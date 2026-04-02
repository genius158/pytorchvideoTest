# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import argparse
import itertools
import logging
import os

import pytorch_lightning
import pytorchvideo.data
import pytorchvideo.models.resnet
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from pytorchvideo.models.x3d import create_x3d, create_x3d_bottleneck_block
from pytorchvideo.models.hub import x3d_xs, x3d_s, x3d_m
from pytorchvideo.layers.swish import Swish
from torch import nn

# 在文件顶部 import 区域添加（约第 15 行附近）
from torchmetrics import Accuracy
from slurm import copy_and_run_with_config
from torch.utils.data import DistributedSampler, RandomSampler
from torchaudio.transforms import MelSpectrogram, Resample
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
)


"""
本视频分类示例演示了如何将 PyTorchVideo 的模型、数据集和变换与 PyTorch Lightning 模块结合使用。具体展示了如何构建一个简单的流水线，在 Kinetics 视频数据集上训练 Resnet。

即使你没有 PyTorch Lightning 的经验也不用担心，示例中会对 Lightning 模块的工作方式进行解释。

代码可以分为三个主要部分：
1. VideoClassificationLightningModule（pytorch_lightning.LightningModule）：定义了
    - 模型如何构建，
    - 训练或验证的内部循环（即从一个小批量计算损失/指标）
    - 优化器配置

2. KineticsDataModule（pytorch_lightning.LightningDataModule）：定义了
    - 如何获取/准备数据集
    - 相关数据集的训练和验证 dataloader

3. pytorch_lightning.Trainer：这是一个具体的 PyTorch Lightning 类，提供训练流水线配置，并通过 fit(<lightning_module>, <data_module>) 启动训练/验证循环。

这三部分在 train() 函数中组合。其余细节将在代码中逐步解释。
"""


class VideoClassificationLightningModule(pytorch_lightning.LightningModule):
    def __init__(self, args):
        """
        该 LightningModule 实现构建了一个 PyTorchVideo 的 X3D 模型，
        定义了训练和验证所用的损失函数（交叉熵），并配置了优化器。
        """
        self.args = args
        super().__init__()
        self.train_accuracy = Accuracy(task="multiclass", num_classes=20)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=20)

        #############
        # PTV 模型 #
        #############

        # 用户提供了 PyTorchVideo 的 X3D 测试文件，我需要根据这个测试文件中的参数来修正 trainx3d.py 中的 create_x3d 调用。 从测试文件中可以看到 create_x3d 的正确参数： 
        # - input_clip_length: 输入视频帧数 
        # - input_crop_size: 输入裁剪尺寸 
        # - model_num_class: 分类类别数 
        # - dropout_rate: dropout 率 
        # - width_factor: 宽度因子 
        # - depth_factor: 深度因子 
        # - norm: 归一化层 
        # - activation: 激活函数 
        # - stem_dim_in: stem 层输入维度 
        # - stem_conv_kernel_size: stem 卷积核大小 
        # - stem_conv_stride: stem 卷积步长 
        # - stage_conv_kernel_size: 各阶段卷积核大小 
        # - stage_spatial_stride: 各阶段空间步长 
        # - stage_temporal_stride: 各阶段时间步长 
        # - bottleneck: bottleneck 块创建函数 
        # - bottleneck_factor: bottleneck 因子（expand_ratio） 
        # - se_ratio: SE 模块比例 - inner_act: 内部激活函数 
        # - head_dim_out: 输出层维度 - head_pool_act: 池化激活 
        # - head_bn_lin5_on: 是否使用 BN 
        # - head_activation: 头部激活函数 X3D 变体参数： 
        # - X3D-XS: (4, 160, 2.0, 2.2, 2.25) 
        # - (input_clip_length, input_crop_size, width_factor, depth_factor, bottleneck_factor) 
        # - X3D-S: (13, 160, 2.0, 2.2, 2.25) 
        # - X3D-M: (16, 224, 2.0, 2.2, 2.25) 
        # - X3D-L: (16, 312, 2.0, 5.0, 2.25) 现在我需要根据这些参数配置来更新 trainx3d.py 文件中的模型创建代码。

        # 使用 X3D 模型 - 根据测试文件的正确参数
        if self.args.arch == "x3d_xs":
            self.model = x3d_xs(pretrained=False, model_num_class=20)
            self.batch_key = "video"
        elif self.args.arch == "x3d_s":
            self.model = x3d_s(pretrained=False, model_num_class=20)
            self.batch_key = "video"
        elif self.args.arch == "x3d_m":
            self.model = x3d_m(pretrained=False, model_num_class=20)
            self.batch_key = "video"
        else:
            raise Exception(f"{self.args.arch} not supported")

    def on_train_epoch_start(self):
        """
        分布式训练时，需要设置数据集的视频采样器的 epoch，以保证正确的随机打乱。
        """
        # epoch = self.trainer.current_epoch
        # # 修复：使用新版本的分布式检测方式
        # if self.trainer.num_devices > 1 or self.trainer.num_nodes > 1:
        #     self.trainer.datamodule.train_dataset.dataset.video_sampler.set_epoch(epoch)

    def forward(self, x):
        """
        前向传播，定义预测/推理操作。
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        该函数在训练 epoch 的内部循环中被调用。它必须返回一个损失值，供 loss.backward() 使用。
        可以用 self.log(...) 记录训练指标。

        PyTorchVideo 的 batch 是一个字典，包含每种模态或元数据。Kinetics 的典型 key 如下：
           {
               'video': <video_tensor>,
               'audio': <audio_tensor>,
               'label': <action_label>,
           }

        - "video"：张量，形状为 (batch, channels, time, height, width)
        - "audio"：张量，形状为 (batch, channels, time, 1, frequency)
        - "label"：张量，形状为 (batch, 1)

        PyTorchVideo 的模型和变换都要求输入字典结构和张量形状一致，因此这里只需解包字典并送入模型/损失函数。
        """
        x = batch[self.batch_key]
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, batch["label"])
        acc = self.train_accuracy(F.softmax(y_hat, dim=-1), batch["label"])
        self.log("train_loss", loss)
        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """
        该函数在验证循环的内部被调用。对于本例来说，与训练循环类似，只是指标名称不同。
        """
        x = batch[self.batch_key]
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, batch["label"])
        acc = self.val_accuracy(F.softmax(y_hat, dim=-1), batch["label"])
        self.log("val_loss", loss)
        self.log(
            "val_acc", acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        return loss

    def configure_optimizers(self):
        """
        使用 SGD 优化器，并配合每步余弦退火学习率调度器。
        """
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.args.max_epochs, last_epoch=-1
        )
        return [optimizer], [scheduler]


class KineticsDataModule(pytorch_lightning.LightningDataModule):
    """
    该 LightningDataModule 实现为训练和验证集构建了 PyTorchVideo 的 Kinetics 数据集。
    定义了每个分区的数据增强和预处理变换，并配置了 PyTorch 的 DataLoader。
    """

    def __init__(self, args):
        self.args = args
        super().__init__()

    def _make_transforms(self, mode: str):
        """
        ##################
        # PTV 变换 #
        ##################

        # 每个 PyTorchVideo 数据集都有一个 "transform" 参数。该参数接收一个 Callable[[Dict], Any]，
        # 用于对数据集输出的字典进行应用特定的处理或增强。变换可以由用户自定义，也可以复用领域库（如视频推荐用 TorchVision，音频推荐用 TorchAudio）。
        #
        # 为了提升不同领域变换库的互操作性，PyTorchVideo 提供了字典变换 API：
        #   - ApplyTransformToKey(key, transform)：对指定模态应用变换
        #   - RemoveKey(key)：移除某个模态
        #
        # 如果推荐库没有提供常用变换，PyTorchVideo 会以相同结构补充。例如 TorchVision 没有 RandomShortSideScale，
        # PyTorchVideo 就补充了该视频变换。
        """
        if self.args.data_type == "video":
            transform = [
                self._video_transform(mode),
                RemoveKey("audio"),
            ]
        elif self.args.data_type == "audio":
            transform = [
                self._audio_transform(),
                RemoveKey("video"),
            ]
        else:
            raise Exception(f"{self.args.data_type} not supported")

        return Compose(transform)

    def _video_transform(self, mode: str):
        """
        该函数展示了如何在同一个 Callable 中结合 PyTorchVideo 和 TorchVision 的变换。
        训练模式下使用带有 "Random" 前缀的数据增强，验证模式下使用确定性变换。
        """
        args = self.args
        return ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(args.video_num_subsampled),
                    Normalize(args.video_means, args.video_stds),
                ]
                + (
                    [
                        RandomShortSideScale(
                            min_size=args.video_min_short_side_scale,
                            max_size=args.video_max_short_side_scale,
                        ),
                        RandomCrop(args.video_crop_size),
                        RandomHorizontalFlip(p=args.video_horizontal_flip_p),
                    ]
                    if mode == "train"
                    else [
                        ShortSideScale(args.video_min_short_side_scale),
                        CenterCrop(args.video_crop_size),
                    ]
                )
            ),
        )

    def _audio_transform(self):
        """
        该函数展示了如何在同一个 Callable 中结合 PyTorchVideo 和 TorchAudio 的变换。
        """
        args = self.args
        n_fft = int(
            float(args.audio_resampled_rate) / 1000 * args.audio_mel_window_size
        )
        hop_length = int(
            float(args.audio_resampled_rate) / 1000 * args.audio_mel_step_size
        )
        eps = 1e-10
        return ApplyTransformToKey(
            key="audio",
            transform=Compose(
                [
                    Resample(
                        orig_freq=args.audio_raw_sample_rate,
                        new_freq=args.audio_resampled_rate,
                    ),
                    MelSpectrogram(
                        sample_rate=args.audio_resampled_rate,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        n_mels=args.audio_num_mels,
                        center=False,
                    ),
                    Lambda(lambda x: x.clamp(min=eps)),
                    Lambda(torch.log),
                    UniformTemporalSubsample(args.audio_mel_num_subsample),
                    Lambda(lambda x: x.transpose(1, 0)),  # (F, T) -> (T, F)
                    Lambda(
                        lambda x: x.view(1, x.size(0), 1, x.size(1))
                    ),  # (T, F) -> (1, T, 1, F)
                    Normalize((args.audio_logmel_mean,), (args.audio_logmel_std,)),
                ]
            ),
        )

    def train_dataloader(self):
        """
        定义 PyTorch Lightning Trainer 用于训练/测试的训练 DataLoader。
        """
            # 修复：使用新版本的分布式检测方式
        sampler = RandomSampler

        train_transform = self._make_transforms(mode="train")
        self.train_dataset = LimitDataset(
            pytorchvideo.data.Kinetics(
                data_path=os.path.join(self.args.data_path, "train.csv"),
                clip_sampler=pytorchvideo.data.make_clip_sampler(
                    "random", self.args.clip_duration
                ),
                video_path_prefix=self.args.video_path_prefix,
                transform=train_transform,
                video_sampler=sampler,
            )
        )
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
        )

    def val_dataloader(self):
        """
        定义 PyTorch Lightning Trainer 用于训练/测试的验证 DataLoader。
        """
        # 修复：使用新版本的分布式检测方式

        sampler = RandomSampler
        val_transform = self._make_transforms(mode="val")

        self.val_dataset = pytorchvideo.data.Kinetics(
            data_path=os.path.join(self.args.data_path, "val.csv"),
            clip_sampler=pytorchvideo.data.make_clip_sampler(
                "uniform", self.args.clip_duration
            ),
            video_path_prefix=self.args.video_path_prefix,
            transform=val_transform,
            video_sampler=sampler,
        )
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
        )


class LimitDataset(torch.utils.data.Dataset):
    """
    为保证每个 epoch 获取的样本数恒定，使用该 LimitDataset 包装器。
    这是因为底层部分视频在抓取或解码时可能损坏，但我们始终希望每个 epoch 步数一致。
    """

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.dataset_iter = itertools.chain.from_iterable(
            itertools.repeat(iter(dataset), 2)
        )

    def __getitem__(self, index):
        return next(self.dataset_iter)

    def __len__(self):
        return self.dataset.num_videos


def main():
    """
    要在 Kinetics 数据集上训练 ResNet，我们构建上述两个模块，并传递给 pytorch_lightning.Trainer 的 fit 函数。

    本示例既可本地运行（使用默认参数），也可在 Slurm 集群上运行。若需在集群上运行，需提供 --on_cluster 参数。
    """
    setup_logger()

    pytorch_lightning.trainer.seed_everything()
    parser = argparse.ArgumentParser()

    #  Cluster parameters.
    parser.add_argument("--on_cluster", action="store_true")
    parser.add_argument("--job_name", default="ptv_video_classification", type=str)
    parser.add_argument("--working_directory", default=".", type=str)
    parser.add_argument("--partition", default="dev", type=str)

    # Model parameters.
    parser.add_argument("--lr", "--learning-rate", default=0.1, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument(
        "--arch",
        default="x3d_xs",
        choices=["x3d_xs", "x3d_s", "x3d_m"],
        type=str,
    )

    # Data parameters.
    parser.add_argument("--data_path", default=None, type=str, required=True)
    parser.add_argument("--video_path_prefix", default="", type=str)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--clip_duration", default=2, type=float)
    parser.add_argument(
        "--data_type", default="video", choices=["video", "audio"], type=str
    )
    parser.add_argument("--video_num_subsampled", default=4, type=int)  # X3D 通常使用 4 帧
    parser.add_argument("--video_crop_size", default=160, type=int)
    parser.add_argument("--video_means", default=(0.45, 0.45, 0.45), type=tuple)  # 新增
    parser.add_argument("--video_stds", default=(0.225, 0.225, 0.225), type=tuple)  # 新增
    parser.add_argument("--video_min_short_side_scale", default=256, type=int)
    parser.add_argument("--video_max_short_side_scale", default=320, type=int)
    parser.add_argument("--video_horizontal_flip_p", default=0.5, type=float)  # 新增
    parser.add_argument("--audio_raw_sample_rate", default=44100, type=int)
    parser.add_argument("--audio_resampled_rate", default=16000, type=int)
    parser.add_argument("--audio_mel_window_size", default=32, type=int)
    parser.add_argument("--audio_mel_step_size", default=16, type=int)
    parser.add_argument("--audio_num_mels", default=80, type=int)
    parser.add_argument("--audio_mel_num_subsample", default=128, type=int)
    parser.add_argument("--audio_logmel_mean", default=-7.03, type=float)
    parser.add_argument("--audio_logmel_std", default=4.66, type=float)

    # Trainer parameters.
    parser = pytorch_lightning.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        max_epochs=5,
        callbacks=[LearningRateMonitor()],
        replace_sampler_ddp=False,
    )

    # Build trainer, ResNet lightning-module and Kinetics data-module.
    args = parser.parse_args()

    if args.on_cluster:
        copy_and_run_with_config(
            train,
            args,
            args.working_directory,
            job_name=args.job_name,
            time="72:00:00",
            partition=args.partition,
            gpus_per_node=args.gpus,
            ntasks_per_node=args.gpus,
            cpus_per_task=10,
            mem="470GB",
            nodes=args.num_nodes,
            constraint="volta32gb",
        )
    else:  # local
        train(args)


def train(args):
    trainer = pytorch_lightning.Trainer.from_argparse_args(args)
    classification_module = VideoClassificationLightningModule(args)
    data_module = KineticsDataModule(args)
    trainer.fit(classification_module, data_module)


def setup_logger():
    ch = logging.StreamHandler()
    formatter = logging.Formatter("\n%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ch.setFormatter(formatter)
    logger = logging.getLogger("pytorchvideo")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(ch)


if __name__ == "__main__":
    main()
