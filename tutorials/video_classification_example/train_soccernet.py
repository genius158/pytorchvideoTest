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
from pathlib import Path

import av
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint

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

    def __init__(self, data_root, clip_duration=4.0, num_frames=8, transform=None):
        self.data_root = Path(data_root)
        self.clip_duration = clip_duration
        self.num_frames = num_frames
        self.transform = transform
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

    def _load_clip(self, video_path, center_sec):
        half = self.clip_duration / 2.0
        start_sec = max(0.0, center_sec - half)
        end_sec = center_sec + half
        frames = []
        try:
            container = av.open(video_path)
            stream = container.streams.video[0]
            seek_ts = int(start_sec / stream.time_base)
            container.seek(seek_ts, stream=stream)
            for frame in container.decode(video=0):
                t = float(frame.pts * stream.time_base)
                if t < start_sec:
                    continue
                if t > end_sec:
                    break
                frames.append(frame.to_rgb().to_ndarray())
            container.close()
        except Exception as e:
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
        clip = self._load_clip(video_path, center_sec)
        sample = {"video": clip, "label": label_idx}
        if self.transform:
            sample = self.transform(sample)
        sample["label"] = torch.tensor(sample["label"], dtype=torch.long)
        return sample


# ==========================================================================
# 2. Transform
# ==========================================================================
def get_video_transform(mode="train"):
    mean = [0.45, 0.45, 0.45]
    std  = [0.225, 0.225, 0.225]
    if mode == "train":
        t = Compose([
            UniformTemporalSubsample(8),
            Lambda(lambda x: x / 255.0),
            Normalize(mean, std),
            RandomShortSideScale(min_size=160, max_size=200),
            RandomCrop(160),
            RandomHorizontalFlip(p=0.5),
        ])
    else:
        t = Compose([
            UniformTemporalSubsample(8),
            Lambda(lambda x: x / 255.0),
            Normalize(mean, std),
            RandomShortSideScale(min_size=160, max_size=160),
            CenterCrop(160),
        ])
    return ApplyTransformToKey(key="video", transform=t)


# ==========================================================================
# 3. LightningModule
# ==========================================================================
class SoccerNetClassifier(pl.LightningModule):
    def __init__(self, num_classes=NUM_CLASSES, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = torch.hub.load(
            'facebookresearch/pytorchvideo', 'x3d_xs', pretrained=True
        )
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
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/acc",  self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["video"], batch["label"]
        logits = self(x)
        loss = self.criterion(logits, y)
        self.val_acc(logits, y)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc",  self.val_acc, prog_bar=True)

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
    p.add_argument("--checkpoint_dir", type=str, default=".checkpoints_soccernet")
    p.add_argument("--num_classes",    type=int,   default=NUM_CLASSES)
    p.add_argument("--clip_duration",  type=float, default=4.0)
    p.add_argument("--num_frames",     type=int,   default=8)
    p.add_argument("--batch_size",     type=int,   default=2)
    p.add_argument("--num_workers",    type=int,   default=8)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--max_epochs",     type=int,   default=50)
    p.add_argument("--accelerator",    type=str,   default="auto",
                   choices=["auto", "cpu", "gpu", "mps"])
    p.add_argument("--resume", action="store_true",
                   help="从 checkpoint_dir/last.ckpt 断点续训")
    return p.parse_args()


def main():
    args = parse_args()

    data_root  = Path(args.data_root)
    train_root = data_root / "train" if (data_root / "train").exists() else data_root
    val_root   = data_root / "valid" if (data_root / "valid").exists() else data_root

    train_ds = SoccerNetClipDataset(
        str(train_root), args.clip_duration, args.num_frames,
        transform=get_video_transform("train"),
    )
    val_ds = SoccerNetClipDataset(
        str(val_root), args.clip_duration, args.num_frames,
        transform=get_video_transform("val"),
    )

    pin = args.accelerator != "cpu"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              num_workers=args.num_workers, shuffle=True,
                              pin_memory=pin, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              num_workers=args.num_workers, shuffle=False,
                              pin_memory=pin)

    model = SoccerNetClassifier(num_classes=args.num_classes, lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="soccernet-{epoch:02d}-{val/acc:.3f}",
        save_last=True,
        monitor="val/acc",
        mode="max",
        save_top_k=3,
    )

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=1,
        max_epochs=args.max_epochs,
        precision=32,
        log_every_n_steps=10,
        callbacks=[ckpt_cb],
    )

    last_ckpt = os.path.join(args.checkpoint_dir, "last.ckpt")
    ckpt_path = last_ckpt if (args.resume and os.path.exists(last_ckpt)) else None
    print(f">>> {'从断点恢复: ' + ckpt_path if ckpt_path else '开始全新训练'}")

    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)

    print(f"\n训练完成！最佳模型: {ckpt_cb.best_model_path}")
    if ckpt_cb.best_model_score is not None:
        print(f"最佳 val/acc: {ckpt_cb.best_model_score:.4f}")


if __name__ == "__main__":
    main()
