import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

# 导入 pytorchvideo 相关组件
from pytorchvideo.data import Ucf101, make_clip_sampler
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    CenterCrop,
)

# --- 1. 定义 LightningModule ---
class VideoClassificationLightningModule(pl.LightningModule):
    def __init__(self, num_classes=20, lr=1e-5):
        super().__init__()
        self.save_hyperparameters()

        # 步骤 A: 加载预训练模型 (X3D-XS 是最适合 CPU 的轻量模型)
        # 使用 torch.hub 确保模型结构和预训练权重匹配
        self.model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_xs', pretrained=True)

        # 步骤 B: 替换分类头 (Final Projection Layer)
        # X3D 的结构中，最后的全连接层位于 blocks[5].proj
        in_features = self.model.blocks[5].proj.in_features
        self.model.blocks[5].proj = nn.Linear(in_features, num_classes)

        # 步骤 C: 评价指标
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # 视频数据在 batch["video"]，标签在 batch["label"]
        x = batch["video"]
        y = batch["label"]

        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = self.accuracy(y_hat, y)

        # 记录日志
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["video"]
        y = batch["label"]
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = self.accuracy(y_hat, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # 使用较小的学习率防止梯度爆炸
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

# --- 2. 定义数据增强 (Transform) ---
def get_video_transform(mode="train"):
    """
    针对视频数据的预处理流水线
    """
    # 归一化标准值 (来自 ImageNet/Kinetics)
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]

    if mode == "train":
        transform = Compose([
            # 时间轴下采样：从视频中均匀抽取 8 帧
            UniformTemporalSubsample(8),
            # 像素归一化 [0, 255] -> [0, 1]
            Lambda(lambda x: x / 255.0),
            Normalize(mean, std),
            # 空间增强
            RandomShortSideScale(min_size=160, max_size=200),
            RandomCrop(160),
            RandomHorizontalFlip(p=0.5),
        ])
    else:
        transform = Compose([
            UniformTemporalSubsample(8),
            Lambda(lambda x: x / 255.0),
            Normalize(mean, std),
            CenterCrop(160),
        ])

    # 包装 Transform，使其仅作用于字典中的 "video" 键
    return ApplyTransformToKey(key="video", transform=transform)

# --- 3. 主训练流程 ---
def main():
    # --- 配置路径 (请根据你的本地环境修改) ---
    # 假设你的目录结构为:
    TRAIN_CSV = "train.csv"
    VAL_CSV = "val.csv"

    print("Checking dataset...")

    # --- 实例化模型 ---
    model = VideoClassificationLightningModule(num_classes=20, lr=1e-5)

    # --- 数据加载器 ---
    train_dataset = Ucf101(
        data_path=TRAIN_CSV,
        clip_sampler=make_clip_sampler("random", 1.0), # 随机截取 2 秒片段
        transform=get_video_transform("train"),
        decode_audio=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,        # CPU 训练建议保持在 1-4 之间
        num_workers=4,       # 根据 CPU 核心数调整，若卡住请设为 0
        pin_memory=False     # CPU 模式下设为 False
    )

    val_dataset = Ucf101(
        data_path=VAL_CSV,
        clip_sampler=make_clip_sampler("uniform", 1.0),
        transform=get_video_transform("val"),
        decode_audio=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        num_workers=4,
        pin_memory=False
    )

    # --- 训练器设置 ---
    trainer = pl.Trainer(
        accelerator="cpu",   # 明确使用 CPU
        devices=1,           # 核心数
        max_epochs=30,       # 建议训练 30 轮以上观察趋势
        precision=32,        # CPU 不支持 16 位精度，必须设为 32
        log_every_n_steps=5, # 频繁记录日志方便观察 Loss
    )

    # 开始执行
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    # 针对 Windows 用户多进程训练的保护
    main()