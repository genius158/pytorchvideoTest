"""
SoccerNet BAS-2025 球动作分类训练脚本 (基于 X3D + PyTorch Lightning)

数据集结构 (SN-BAS-2025):
  <data_root>/
    train/
      england_efl/2019-2020/<game>/
        ├── 224p.mp4          # 全场视频（单文件）
        ├── 720p.mp4
        └── Labels-ball.json  # 球动作标注
    valid/
      ...

Labels-ball.json 格式:
  {
    "halftime": "1 - 47:16",
    "annotations": [
      { "gameTime": "1 - 00:01", "label": "PASS",
        "position": "1120",      <- 毫秒
        "team": "left", "visibility": "visible" }
    ]
  }

使用方式:
    # 训练（data_root 指向包含 train/valid 子目录的根目录）
    python train_soccernet.py --data_root /path/to/SN-BAS-2025

    # GPU 训练
    python train_soccernet.py --data_root /path/to/SN-BAS-2025 --accelerator gpu --batch_size 16

    # 断点续训
    python train_soccernet.py --data_root /path/to/SN-BAS-2025 --resume
"""

import os
import json
import argparse
import random
from collections import OrderedDict
from pathlib import Path

import av
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    UniformTemporalSubsample,
)
from pytorchvideo.models.hub import x3d_xs
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    CenterCrop,
)

# ------------------------------------------------------------------
# 与 TorchVideo Android Demo 对齐的输入配置
#   Constants.java:
#   - COUNT_OF_FRAMES_PER_INFERENCE = 4（此脚本当前改为 8，用于 X3D-XS）
#   - TARGET_VIDEO_SIZE = 160
#   - MEAN_RGB = [0.45, 0.45, 0.45]
#   - STD_RGB  = [0.225, 0.225, 0.225]
# ------------------------------------------------------------------
COUNT_OF_FRAMES_PER_INFERENCE = 8
TARGET_VIDEO_SIZE = 160
MEAN_RGB = [0.45, 0.45, 0.45]
STD_RGB = [0.225, 0.225, 0.225]

# ------------------------------------------------------------------
# SN-BAS-2025 的 12 个球动作类别
# ------------------------------------------------------------------
SOCCERNET_ACTIONS = [
    "PASS",                     # 0
    "DRIVE",                    # 1
    "HIGH PASS",                # 2
    "HEADER",                   # 3
    "SHOT",                     # 4
    "CROSS",                    # 5
    "THROW IN",                 # 6
    "FREE KICK",                # 7
    "GOAL",                     # 8
    "OUT",                      # 9
    "PLAYER SUCCESSFUL TACKLE", # 10
    "BALL PLAYER BLOCK",        # 11
]

ACTION_TO_IDX = {action: idx for idx, action in enumerate(SOCCERNET_ACTIONS)}
NUM_CLASSES = len(SOCCERNET_ACTIONS)


# ==========================================================================
# 1. Dataset
# ==========================================================================
class SoccerNetClipDataset(Dataset):
    """
    从 SN-BAS-2025 的 Labels-ball.json 中提取动作片段。
    每个样本是以 position（毫秒）为中心、固定时长的视频片段。

    Args:
        data_root     : split 目录，如 .../SN-BAS-2025/train
        clip_duration : 每个片段时长（秒），默认 4.0
        num_frames    : 均匀采样帧数，默认 8
        transform     : pytorchvideo/torchvision 变换
    """

    def __init__(
        self,
        data_root,
        clip_duration=4.0,
        num_frames=8,
        transform=None,
        time_jitter_sec=0.0,
    ):
        self.data_root = Path(data_root)
        self.clip_duration = clip_duration
        self.num_frames = num_frames
        self.transform = transform
        self.time_jitter_sec = max(0.0, float(time_jitter_sec))
        self._container_cache = OrderedDict()
        self._max_open_videos = 8
        self.samples = self._build_sample_list()
        print(f"[SoccerNetClipDataset] {data_root} -> {len(self.samples)} samples")

    def _build_sample_list(self):
        samples = []
        label_files = sorted(self.data_root.rglob("Labels-ball.json"))
        if not label_files:
            raise FileNotFoundError(
                f"在 {self.data_root} 下未找到任何 Labels-ball.json。\n"
                "请确认 --data_root 指向正确的 split 目录，例如：\n"
                "  --data_root /path/to/SN-BAS-2025/train"
            )

        for label_path in label_files:
            game_dir = label_path.parent
            video_file = game_dir / "224p.mp4"
            if not video_file.exists():
                video_file = game_dir / "720p.mp4"
            if not video_file.exists():
                continue

            with open(label_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for ann in data.get("annotations", []):
                label = ann.get("label", "")
                if label not in ACTION_TO_IDX:
                    continue
                try:
                    t_sec = int(ann["position"]) / 1000.0
                except (KeyError, ValueError):
                    continue
                samples.append((str(video_file), t_sec, ACTION_TO_IDX[label]))

        if not samples:
            raise RuntimeError("未能提取任何有效样本，请检查视频文件是否存在。")
        return samples

    def _get_video_container(self, video_path):
        if video_path in self._container_cache:
            container, stream = self._container_cache.pop(video_path)
            self._container_cache[video_path] = (container, stream)
            return container, stream

        container = av.open(video_path)
        stream = container.streams.video[0]
        self._container_cache[video_path] = (container, stream)

        if len(self._container_cache) > self._max_open_videos:
            _, (old_container, _) = self._container_cache.popitem(last=False)
            old_container.close()

        return container, stream

    def _load_clip(self, video_path, center_sec):
        half = self.clip_duration / 2.0
        start_sec = max(0.0, center_sec - half)
        end_sec = center_sec + half
        frames = []
        try:
            container, stream = self._get_video_container(video_path)
            seek_ts = int(start_sec / stream.time_base)
            container.seek(seek_ts, stream=stream)
            for frame in container.decode(video=0):
                t = float(frame.pts * stream.time_base)
                if t < start_sec:
                    continue
                if t > end_sec:
                    break
                frames.append(frame.to_rgb().to_ndarray())
        except Exception as e:
            cached = self._container_cache.pop(video_path, None)
            if cached is not None:
                cached[0].close()
            print(f"[警告] {video_path} @ {center_sec:.1f}s : {e}")

        if not frames:
            frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * self.num_frames

        indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
        clip = np.stack([frames[i] for i in indices], axis=0)   # [T, H, W, C]
        return torch.from_numpy(clip).float().permute(3, 0, 1, 2)  # [C, T, H, W]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, center_sec, label_idx = self.samples[idx]
        if self.time_jitter_sec > 0:
            center_sec = max(0.0, center_sec + random.uniform(-self.time_jitter_sec, self.time_jitter_sec))
        clip = self._load_clip(video_path, center_sec)
        sample = {"video": clip, "label": label_idx}
        if self.transform:
            sample = self.transform(sample)
        sample["label"] = torch.tensor(sample["label"], dtype=torch.long)
        return sample

    def __del__(self):
        for container, _ in self._container_cache.values():
            container.close()
        self._container_cache.clear()


# ==========================================================================
# 2. Transform
# ==========================================================================
def temporal_stride_sample(video: torch.Tensor, num_frames: int, stride: int) -> torch.Tensor:
    """
    在时间维按固定步长采样，输入/输出形状均为 [C, T, H, W]。
    - 优先使用固定 stride 的连续索引（居中窗口）
    - 当帧数不足时回退到等间隔补齐到 num_frames
    """
    t = int(video.shape[1])
    if t <= 0:
        return video

    need = (num_frames - 1) * stride + 1
    if t >= need:
        start = (t - need) // 2
        idx = start + torch.arange(num_frames) * stride
    else:
        idx = torch.linspace(0, t - 1, steps=num_frames).long()

    idx = idx.to(device=video.device)
    return video.index_select(1, idx)


def get_video_transform(mode="train", num_frames=COUNT_OF_FRAMES_PER_INFERENCE, temporal_stride=12):
    mean = MEAN_RGB
    std  = STD_RGB
    if mode == "train":
        t = Compose([
            Lambda(lambda x: temporal_stride_sample(x, num_frames=num_frames, stride=temporal_stride)),
            Lambda(lambda x: x / 255.0),
            Normalize(mean, std),
            RandomShortSideScale(min_size=TARGET_VIDEO_SIZE, max_size=TARGET_VIDEO_SIZE),
            RandomCrop(TARGET_VIDEO_SIZE),
            RandomHorizontalFlip(p=0.5),
        ])
    else:
        t = Compose([
            Lambda(lambda x: temporal_stride_sample(x, num_frames=num_frames, stride=temporal_stride)),
            Lambda(lambda x: x / 255.0),
            Normalize(mean, std),
            RandomShortSideScale(min_size=TARGET_VIDEO_SIZE, max_size=TARGET_VIDEO_SIZE),
            CenterCrop(TARGET_VIDEO_SIZE),
        ])
    return ApplyTransformToKey(key="video", transform=t)


def get_runtime_config(requested_accelerator: str, requested_devices: int, requested_precision: str):
    def resolve_precision(default_precision):
        if requested_precision == "auto":
            return default_precision
        if requested_precision == "16":
            return 16
        if requested_precision == "32":
            return 32
        if requested_precision == "bf16":
            return "bf16"
        raise ValueError(f"不支持的 precision: {requested_precision}")

    if requested_accelerator == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("指定了 --accelerator gpu，但当前环境未检测到可用 CUDA 设备。")
        return {
            "accelerator": "gpu",
            "devices": requested_devices,
            "precision": resolve_precision(16),
            "pin_memory": True,
            "device_name": torch.cuda.get_device_name(0),
        }

    if requested_accelerator == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise RuntimeError("指定了 --accelerator mps，但当前环境未检测到可用 MPS 设备。")
        return {
            "accelerator": "mps",
            "devices": 1,
            "precision": resolve_precision(32),
            "pin_memory": False,
            "device_name": "Apple Silicon MPS",
        }

    if requested_accelerator == "cpu":
        return {
            "accelerator": "cpu",
            "devices": 1,
            "precision": resolve_precision(32),
            "pin_memory": False,
            "device_name": "CPU",
        }

    if torch.cuda.is_available():
        return {
            "accelerator": "gpu",
            "devices": requested_devices,
            "precision": resolve_precision(16),
            "pin_memory": True,
            "device_name": torch.cuda.get_device_name(0),
        }

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return {
            "accelerator": "mps",
            "devices": 1,
            "precision": resolve_precision(32),
            "pin_memory": False,
            "device_name": "Apple Silicon MPS",
        }

    return {
        "accelerator": "cpu",
        "devices": 1,
        "precision": resolve_precision(32),
        "pin_memory": False,
        "device_name": "CPU",
    }


# ==========================================================================
# 3. LightningModule
# ==========================================================================
class SoccerNetClassifier(pl.LightningModule):
    def __init__(self, num_classes=NUM_CLASSES, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        checkpoint_path = Path(__file__).with_name("X3D_XS.pyth")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"未找到本地预训练权重: {checkpoint_path}")

        self.model = x3d_xs(pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model_state"])
        in_features = self.model.blocks[5].proj.in_features
        self.model.blocks[5].proj = nn.Linear(in_features, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc   = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["video"], batch["label"]
        logits = self(x)
        loss = self.criterion(logits, y)
        self.train_acc(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["video"], batch["label"]
        logits = self(x)
        loss = self.criterion(logits, y)
        self.val_acc(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=1e-6
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


# ==========================================================================
# 4. 主流程
# ==========================================================================
def parse_args():
    p = argparse.ArgumentParser(description="SoccerNet BAS-2025 动作分类训练")
    p.add_argument("--data_root", type=str, required=True,
                   help="数据集根目录（含 train/ 和 valid/ 子目录）")
    p.add_argument("--checkpoint_dir", type=str, default=".checkpoints_soccernet_v4")
    p.add_argument("--num_classes",    type=int,   default=NUM_CLASSES)
    p.add_argument("--clip_duration",  type=float, default=None,
                   help="每个片段时长（秒）。默认自动按 num_frames*stride/fps 计算")
    p.add_argument("--train_time_jitter_sec", type=float, default=0.5,
                   help="训练集时间抖动范围（秒），按 ±jitter 随机偏移事件中心，验证集固定为 0")
    p.add_argument("--num_frames",     type=int,   default=COUNT_OF_FRAMES_PER_INFERENCE,
                   help=f"输入帧数，默认与 Android Demo 对齐为 {COUNT_OF_FRAMES_PER_INFERENCE}")
    p.add_argument("--sampling_stride", type=int, default=6,
                   help="时间采样步长 stride，默认 6")
    p.add_argument("--sampling_fps", type=float, default=30.0,
                   help="用于估算覆盖时长的采样 fps，默认 30")
    p.add_argument("--batch_size",     type=int,   default=4)
    p.add_argument("--num_workers",    type=int,   default=16)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--max_epochs",     type=int,   default=60)
    p.add_argument("--patience",       type=int,   default=30,
                   help="EarlyStopping 容忍轮数，默认 10")
    p.add_argument("--devices",        type=int,   default=1,
                   help="使用的设备数量；GPU 模式下可大于 1，默认 1")
    p.add_argument("--precision",      type=str,   default="auto",
                   choices=["auto", "16", "32", "bf16"],
                   help="训练精度；auto 表示 GPU 默认 16，CPU/MPS 默认 32")
    p.add_argument("--accelerator",    type=str,   default="auto",
                   choices=["auto", "cpu", "gpu", "mps"])
    p.add_argument("--resume", action="store_true",
                   help="从 checkpoint_dir/last.ckpt 断点续训")
    return p.parse_args()


def main():
    args = parse_args()
    if args.clip_duration is None:
        args.clip_duration = args.num_frames * args.sampling_stride / args.sampling_fps

    runtime = get_runtime_config(args.accelerator, args.devices, args.precision)

    if runtime["accelerator"] == "gpu":
        torch.backends.cudnn.benchmark = True

    print(
        f">>> 当前训练设备: {runtime['device_name']} "
        f"(accelerator={runtime['accelerator']}, devices={runtime['devices']}, precision={runtime['precision']})"
    )
    print(
        f">>> 输入配置: frames={args.num_frames}, size={TARGET_VIDEO_SIZE}, "
        f"mean={MEAN_RGB}, std={STD_RGB}"
    )
    print(
        f">>> 时序采样: stride={args.sampling_stride}, fps={args.sampling_fps}, "
        f"覆盖时长≈{args.num_frames * args.sampling_stride / args.sampling_fps:.3f}s, "
        f"clip_duration={args.clip_duration:.3f}s"
    )

    data_root  = Path(args.data_root)
    train_root = data_root / "train" if (data_root / "train").exists() else data_root
    val_root   = data_root / "valid" if (data_root / "valid").exists() else data_root

    train_ds = SoccerNetClipDataset(
        str(train_root), args.clip_duration, args.num_frames,
        transform=get_video_transform("train", num_frames=args.num_frames, temporal_stride=args.sampling_stride),
        time_jitter_sec=args.train_time_jitter_sec,
    )
    val_ds = SoccerNetClipDataset(
        str(val_root), args.clip_duration, args.num_frames,
        transform=get_video_transform("val", num_frames=args.num_frames, temporal_stride=args.sampling_stride),
        time_jitter_sec=0.0,
    )

    pin = runtime["pin_memory"]
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": pin,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **loader_kwargs,
    )

    model = SoccerNetClassifier(num_classes=args.num_classes, lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="soccernet-{epoch:02d}-{val_acc:.3f}",
        save_last=True,
        monitor="val_acc",
        mode="max",
        save_top_k=3,
    )

    early_stop_cb = EarlyStopping(
        monitor="val_acc",
        mode="max",
        patience=args.patience,
        min_delta=0.001,
        verbose=True,
    )

    trainer = pl.Trainer(
        accelerator=runtime["accelerator"],
        devices=runtime["devices"],
        max_epochs=args.max_epochs,
        precision=runtime["precision"],
        log_every_n_steps=10,
        benchmark=(runtime["accelerator"] == "gpu"),
        callbacks=[ckpt_cb, early_stop_cb],
    )

    last_ckpt = os.path.join(args.checkpoint_dir, "last.ckpt")
    ckpt_path = last_ckpt if (args.resume and os.path.exists(last_ckpt)) else None
    print(f">>> {'从断点恢复: ' + ckpt_path if ckpt_path else '开始全新训练'}")

    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

    print("\n训练完成！")
    print(f"训练集样本数: {len(train_ds)}")
    print(f"验证集样本数: {len(val_ds)}")
    print(f"最佳模型: {ckpt_cb.best_model_path}")
    if ckpt_cb.best_model_score is not None:
        print(f"最佳 val_acc: {ckpt_cb.best_model_score:.4f}")
    print(f"最近检查点: {last_ckpt if os.path.exists(last_ckpt) else '未生成'}")


if __name__ == "__main__":
    main()
