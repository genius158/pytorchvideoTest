# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# pyre-strict

from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    TypedDict,
    Union,
)

import pytorch_lightning as pl
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr

# @manual "fbsource//third-party/pypi/omegaconf:omegaconf"
from omegaconf import MISSING, OmegaConf
from pytorch_lightning.utilities import rank_zero_info
from pytorchvideo_trainer.datamodule.transforms import MixVideoBatchWrapper
from pytorchvideo_trainer.module.lr_policy import get_epoch_lr, LRSchedulerConf, set_lr
from pytorchvideo_trainer.module.optimizer import construct_optimizer
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler
from torchrecipes.core.conf import ModuleConf
from torchrecipes.utils.config_utils import get_class_name_str


class Batch(TypedDict):
    """
    PyTorchVideo 的 batch 是包含每种模态或元数据的字典。
    对于 Kinetics，包含如下 key 和类型。
    """

    video: torch.Tensor  # (B, C, T, H, W)
    audio: torch.Tensor  # (B, S)
    label: torch.Tensor  # (B, 1)
    video_index: List[int]  # len(video_index) == B


BatchKey = Literal["video", "audio", "label", "video_index"]
EnsembleMethod = Literal["sum", "max"]


class VideoClassificationModule(pl.LightningModule):
    """
    支持视频分类任务的 Lightning 模块。

    参数说明：
        model (OmegaConf)：用于初始化神经网络模型的 OmegaConf 对象。
            示例配置见 `pytorchvideo_trainer/conf/module/model`
        loss (OmegaConf)：用于初始化损失函数的 OmegaConf 对象。
            示例配置见 `pytorchvideo_trainer/conf/module/loss`
        optim (OmegaConf)：用于构建优化器的 OmegaConf 对象。
            配置 schema 见 `pytorchvideo_trainer.module.optimizer.OptimizerConf`。
            示例配置见 `pytorchvideo_trainer/conf/module/optim`
        metrics (OmegaConf)：要追踪的指标，训练、验证和测试都会用到。
            示例配置见 `pytorchvideo_trainer/conf/module/metricx`
        lr_scheduler (OmegaConf)：训练过程中使用的学习率调度器配置。
            配置 schema 见 `pytorchvideo_trainer.module.lr_policy.LRSchedulerConf`。
            示例配置见 `pytorchvideo_trainer/conf/module/lr_scheduler`
        modality_key (str)：数据处理时使用的模态 key，默认 "video"。
        ensemble_method (str)：测试时视频级别集成方法，可选 ["sum", "max", None]。
            None 表示不做集成。
        num_classes (int)：数据集类别数。
        num_sync_devices (int)：同步 BatchNorm 的 GPU 数，仅在 trainer 的 sync_batchnorm 为 false 时有效。
        batch_transform (OmegaConf)：可选，对整个 mini batch 进行变换的方法配置，如 MixVideo 等。
        clip_gradient_norm (float)：若大于 0，则进行梯度裁剪。由于采用 Lightning 的手动优化，梯度裁剪需在模块内设置。
    """

    def __init__(
        self,
        model: Any,  # pyre-ignore[2]
        loss: Any,  # pyre-ignore[2]
        optim: Any,  # pyre-ignore[2]
        metrics: List[Any],  # pyre-ignore[2]
        lr_scheduler: Optional[Any] = None,  # pyre-ignore[2]
        modality_key: BatchKey = "video",
        ensemble_method: Optional[EnsembleMethod] = None,
        num_classes: int = 400,
        num_sync_devices: int = 1,
        batch_transform: Optional[Any] = None,  # pyre-ignore[2]
        clip_gradient_norm: float = 0.0,
    ) -> None:
        super().__init__()
        self.automatic_optimization = False

        self.model: nn.Module = instantiate(model, _convert_="all")
        self.loss: nn.Module = instantiate(loss)
        self.batch_transform = instantiate(batch_transform)  # pyre-ignore[4]
        rank_zero_info(OmegaConf.to_yaml(optim))
        self.optim: torch.optim.Optimizer = construct_optimizer(self.model, optim)
        self.lr_scheduler_conf: LRSchedulerConf = lr_scheduler
        self.modality_key: BatchKey = modality_key
        self.ensemble_method: Optional[EnsembleMethod] = ensemble_method
        self.num_classes: int = num_classes
        self.clip_gradient_norm = clip_gradient_norm

        self.metrics: Mapping[str, nn.Module] = {
            metric_conf.name: instantiate(metric_conf.config) for metric_conf in metrics
        }

        self.train_metrics: nn.ModuleDict = nn.ModuleDict()
        self.val_metrics: nn.ModuleDict = nn.ModuleDict()
        self.test_metrics: nn.ModuleDict = nn.ModuleDict()

        self.save_hyperparameters()

        # These are used for data ensembling in the test stage.
        self.video_preds: Dict[int, torch.Tensor] = {}
        self.video_labels: Dict[int, torch.Tensor] = {}
        self.video_clips_cnts: Dict[int, int] = {}

        # Sync BatchNorm
        self.num_sync_devices = num_sync_devices

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit":
            self.train_metrics.update(self.metrics)
            self.val_metrics.update(self.metrics)
        else:
            self.test_metrics.update(self.metrics)

    # pyre-ignore[14]: *args, **kwargs are not torchscriptable.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，定义预测/推理操作。
        """
        return self.model(x)

    def _num_training_steps_per_epoch(self) -> int:
        """根据 datamodule 和设备数推断每个 epoch 的训练步数。"""
        dataloader = self.trainer.datamodule.train_dataloader()
        world_size = self.trainer.world_size

        # TODO: Make sure other dataloaders has this property
        dataset_size = self.trainer.limit_train_batches
        dataset_size *= len(dataloader.dataset._labeled_videos)

        # TODO: Make sure other dataloaders has this property
        return dataset_size // world_size // dataloader.batch_size

    def manual_update_lr(self) -> None:
        """手动更新优化器学习率的工具函数"""

        opt = self.optimizers()

        if self.lr_scheduler_conf is not None:
            # pyre-ignore[6]
            exact_epoch = float(self.cur_epoch_step) / float(
                self._num_training_steps_per_epoch()
            )
            exact_epoch += self.trainer.current_epoch
            lr = get_epoch_lr(exact_epoch, self.lr_scheduler_conf)
            self.log("LR", lr, on_step=True, prog_bar=True)
            self.log("ExactE", exact_epoch, on_step=True, prog_bar=True)

            if isinstance(opt, list):
                for op in opt:
                    set_lr(op, lr)  # pyre-ignore[6]
            else:
                set_lr(opt, lr)  # pyre-ignore[6]

    def manual_zero_opt_grad(self) -> None:
        """手动将优化器梯度清零的工具函数"""
        opt = self.optimizers()
        if isinstance(opt, list):
            for op in opt:
                op.zero_grad()  # pyre-ignore[16]
        else:
            opt.zero_grad()

    def manual_opt_step(self) -> None:
        """手动执行优化器 step 的工具函数"""
        opt = self.optimizers()
        if isinstance(opt, list):
            for op in opt:
                op.step()
        else:
            opt.step()

    def training_step(
        self, batch: Batch, batch_idx: int, *args: Any, **kwargs: Any
    ) -> None:
        """
        PyTorchVideo 的模型和变换都要求输入字典结构和张量形状一致，因此这里只需解包字典并送入模型/损失函数。
        """
        self.cur_epoch_step += 1  # pyre-ignore[16]

        if self.batch_transform is not None:
            batch = self.batch_transform(batch)

        self.manual_zero_opt_grad()
        self.manual_update_lr()

        # Forward/backward
        loss = self._step(batch, batch_idx, "train")
        self.manual_backward(loss)  # pyre-ignore[6]
        if self.clip_gradient_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip_gradient_norm
            )
        self.manual_opt_step()

    def validation_step(
        self, batch: Batch, batch_idx: int, *args: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        """
        处理验证集的单个 batch。
        """
        return self._step(batch, batch_idx, "val")

    def test_step(
        self, batch: Batch, batch_idx: int, *args: Any, **kwargs: Any
    ) -> Optional[Dict[str, Any]]:
        """
        处理测试集的单个 batch。
        """
        if self.ensemble_method:
            self._test_step_with_data_ensembling(batch, batch_idx)
        else:
            return self._step(batch, batch_idx, "test")

    def _test_step_with_data_ensembling(self, batch: Batch, batch_idx: int) -> None:
        """
        处理测试集的单个 batch。
        """
        assert (
            isinstance(batch, dict)
            and self.modality_key in batch
            and "label" in batch
            and "video_index" in batch
        ), (
            f"Returned batch [{batch}] is not a map with '{self.modality_key}' and"
            + "'label' and 'video_index' keys"
        )

        y_hat = self(batch[self.modality_key])
        preds = torch.nn.functional.softmax(y_hat, dim=-1)
        labels = batch["label"]
        video_ids = torch.tensor(batch["video_index"], device=self.device)

        self._ensemble_at_video_level(preds, labels, video_ids)

    def on_train_epoch_start(self) -> None:
        self._reset_metrics("train")
        self.cur_epoch_step = 0.0  # pyre-ignore[16]

    def on_validation_epoch_start(self) -> None:
        self._reset_metrics("val")

    def on_test_epoch_start(self) -> None:
        self._reset_metrics("test")

    def on_test_epoch_end(self) -> None:
        """Pytorch-Lightning's method for aggregating test metrics at the end of epoch"""
        if self.ensemble_method:
            for video_id in self.video_preds:
                self.video_preds[video_id] = (
                    self.video_preds[video_id] / self.video_clips_cnts[video_id]
                )
            video_preds = torch.stack(list(self.video_preds.values()), dim=0)
            video_labels = torch.tensor(
                list(self.video_labels.values()),
                device=self.device,
            )
            metrics_result = self._compute_metrics(video_preds, video_labels, "test")
            self.log_dict(metrics_result)

    def _ensemble_at_video_level(
        self, preds: torch.Tensor, labels: torch.Tensor, video_ids: torch.Tensor
    ) -> None:
        """
        对同一视频的多个视角预测结果进行集成。
        依赖于 dataloader 会读取同一视频在不同空间裁剪下的多个片段。
        """
        for i in range(preds.shape[0]):
            vid_id = int(video_ids[i])
            self.video_labels[vid_id] = labels[i]
            if vid_id not in self.video_preds:
                self.video_preds[vid_id] = torch.zeros(
                    (self.num_classes), device=self.device, dtype=preds.dtype
                )
                self.video_clips_cnts[vid_id] = 0

            if self.ensemble_method == "sum":
                self.video_preds[vid_id] += preds[i]
            elif self.ensemble_method == "max":
                self.video_preds[vid_id] = torch.max(self.video_preds[vid_id], preds[i])
            self.video_clips_cnts[vid_id] += 1

    def configure_optimizers(
        self,
    ) -> Union[
        torch.optim.Optimizer,
        Tuple[Iterable[torch.optim.Optimizer], Iterable[_LRScheduler]],
    ]:
        """Pytorch-Lightning 配置优化器的方法"""
        return self.optim

    def _step(self, batch: Batch, batch_idx: int, phase_type: str) -> Dict[str, Any]:
        assert (
            isinstance(batch, dict) and self.modality_key in batch and "label" in batch
        ), (
            f"返回的 batch [{batch}] 不是包含 '{self.modality_key}' 和 'label' 键的字典"
        )

        y_hat = self(batch[self.modality_key])
        if phase_type == "train":
            loss = self.loss(y_hat, batch["label"])
            self.log(
                f"Losses/{phase_type}_loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
        else:
            loss = None

        ## TODO: 将 MixUP 变换的指标计算移到单独方法。
        if (
            phase_type == "train"
            and self.batch_transform is not None
            and isinstance(self.batch_transform, MixVideoBatchWrapper)
        ):
            _top_max_k_vals, top_max_k_inds = torch.topk(
                batch["label"], 2, dim=1, largest=True, sorted=True
            )
            idx_top1 = torch.arange(batch["label"].shape[0]), top_max_k_inds[:, 0]
            idx_top2 = torch.arange(batch["label"].shape[0]), top_max_k_inds[:, 1]
            y_hat = y_hat.detach()
            y_hat[idx_top1] += y_hat[idx_top2]
            y_hat[idx_top2] = 0.0
            batch["label"] = top_max_k_inds[:, 0]

        pred = torch.nn.functional.softmax(y_hat, dim=-1)
        metrics_result = self._compute_metrics(pred, batch["label"], phase_type)
        self.log_dict(metrics_result, on_epoch=True)

        return loss

    def _compute_metrics(
        self, pred: torch.Tensor, label: torch.Tensor, phase_type: str
    ) -> Dict[str, torch.Tensor]:
        metrics_dict = getattr(self, f"{phase_type}_metrics")
        metrics_result = {}
        for name, metric in metrics_dict.items():
            metrics_result[f"Metrics/{phase_type}/{name}"] = metric(pred, label)
        return metrics_result

    def _reset_metrics(self, phase_type: str) -> None:
        metrics_dict = getattr(self, f"{phase_type}_metrics")
        for _, metric in metrics_dict.items():
            metric.reset()

    def _convert_to_sync_bn(self) -> None:
        """
        将 BatchNorm 转换为同步 BatchNorm。
        如果 trainer 的 sync_batchnorm 参数为 true，则在所有节点和 GPU 上执行全局同步 BatchNorm。
        否则，在指定数量的 GPU 上执行本地同步 BatchNorm。
        """
        if (
            hasattr(self.trainer.training_type_plugin, "sync_batchnorm")
            and self.trainer.training_type_plugin.sync_batchnorm
        ):
            print("Using Global Synch BatchNorm.")
            return None

        if self.num_sync_devices > 1:
            print(f"Using local Synch BatchNorm over {self.num_sync_devices} devices.")
            pg = create_syncbn_process_group(self.num_sync_devices)
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                self.model, process_group=pg
            )

    def on_fit_start(self) -> None:
        """
        在 fit 开始时调用。
        如果是 DDP，每个进程都会调用。
        """
        self._convert_to_sync_bn()


def create_syncbn_process_group(group_size: int) -> List[int]:
    """
    创建用于同步 BatchNorm 的进程组，组大小为 group_size，并返回当前 GPU 所在的进程组。

    参数：
        group_size (int)：参与同步 BN 的 GPU 数，需 >=2，否则不做处理。
    """
    assert group_size > 1, (
        f"Invalid group size {group_size} to convert to sync batchnorm."
    )

    world_size = torch.distributed.get_world_size()
    assert world_size >= group_size
    assert world_size % group_size == 0

    group = None
    for group_num in range(world_size // group_size):
        group_ids = range(group_num * group_size, (group_num + 1) * group_size)
        cur_group = torch.distributed.new_group(ranks=group_ids)
        if torch.distributed.get_rank() // group_size == group_num:
            group = cur_group
            # can not drop out and return here,
            # every process must go through creation of all subgroups

    assert group is not None
    return group


@dataclass
class VideoClassificationModuleConf(ModuleConf):
    _target_: str = get_class_name_str(VideoClassificationModule)
    model: Any = MISSING  # pyre-ignore[4]
    loss: Any = MISSING  # pyre-ignore[4]
    optim: Any = MISSING  # pyre-ignore[4]
    metrics: List[Any] = MISSING  # pyre-ignore[4]
    lr_scheduler: Optional[Any] = None  # pyre-ignore[4]
    modality_key: str = "video"
    ensemble_method: Optional[str] = None
    num_classes: int = 400
    num_sync_devices: Optional[int] = 1


@dataclass
class VideoClassificationModuleConfVisionTransformer(VideoClassificationModuleConf):
    batch_transform: Optional[Any] = None  # pyre-ignore[4]
    clip_gradient_norm: float = 0.0


cs = ConfigStore()
cs.store(
    group="schema/module",
    name="video_classification_module_conf",
    node=VideoClassificationModuleConf,
    package="module",
)

cs.store(
    group="schema/module",
    name="video_classification_module_conf_vision_transformer",
    node=VideoClassificationModuleConfVisionTransformer,
    package="module",
)


def create_classification_model_from_modelzoo(
    checkpoint_path: str,
    model: nn.Module,
) -> nn.Module:
    """
    从 PyTorchVideo 的模型库 checkpoint 构建模型。

    示例配置见：
    `pytorchvideo_trainer/conf/module/model/from_model_zoo_checkpoint.yaml`

    参数：
        checkpoint_path (str)：预训练模型权重路径。
        model (nn.Module)：要加载权重的模型。
    返回：
        model (nn.Module)：加载了预训练权重的模型。
    """

    with g_pathmgr.open(checkpoint_path, "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")
    state_dict = checkpoint["model_state"]
    model.load_state_dict(state_dict)
    return model


def create_classification_model_from_lightning(
    checkpoint_path: str,
) -> nn.Module:
    """
    从 pytorchvideo_trainer 的 PytorchLightning checkpoint 构建模型。

    示例配置见：
    `pytorchvideo_trainer/conf/module/model/from_lightning_checkpoint.yaml`

    参数：
        checkpoint_path (str)：预训练模型权重路径。
    返回：
        model (nn.Module)：加载了预训练权重的模型。
    """
    lightning_model = VideoClassificationModule.load_from_checkpoint(checkpoint_path)
    return lightning_model.model
