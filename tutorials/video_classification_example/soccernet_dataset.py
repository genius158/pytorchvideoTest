"""
SoccerNet BAS-2025 球动作数据集管理模块

参考 PyTorchVideo 的 Ucf101 设计，提供便利函数来创建 SoccerNet 数据加载器。
"""

from typing import Any, Callable, Dict, Optional
from pathlib import Path

import torch
import torch.nn as nn
import json
import numpy as np
import av
from torch.utils.data import Dataset


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
# SoccerNetClipDataset：核心数据集类
# ==========================================================================
class SoccerNetClipDataset(Dataset):
    """
    从 SN-BAS-2025 的 Labels-ball.json 中提取动作片段。
    每个样本是以 position（毫秒）为中心、固定时长的视频片段。

    Args:
        data_path     : split 目录，如 .../SN-BAS-2025/train 或 .../train/england_efl/2019-2020
        clip_duration : 每个片段时长（秒），默认 4.0
        num_frames    : 均匀采样帧数，默认 8
        transform     : pytorchvideo/torchvision 变换
    """

    def __init__(
        self,
        data_path: str,
        clip_duration: float = 4.0,
        num_frames: int = 8,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ):
        self.data_path = Path(data_path)
        self.clip_duration = clip_duration
        self.num_frames = num_frames
        self.transform = transform
        self.samples = self._build_sample_list()
        print(f"[SoccerNetClipDataset] {data_path}")
        print(f"                      -> {len(self.samples)} samples loaded")

    def _build_sample_list(self):
        """遍历目录，收集 (video_path, timestamp_sec, label_idx) 元组"""
        samples = []
        label_files = sorted(self.data_path.rglob("Labels-ball.json"))

        if not label_files:
            raise FileNotFoundError(
                f"在 {self.data_path} 下未找到任何 Labels-ball.json。\n"
                f"请确认数据路径正确，例如：\n"
                f"  - {self.data_path}/train\n"
                f"  - {self.data_path}/valid"
            )

        game_count = 0
        for label_path in label_files:
            game_dir = label_path.parent

            # 优先用 224p.mp4，备选 720p.mp4
            video_file = game_dir / "224p.mp4"
            if not video_file.exists():
                video_file = game_dir / "720p.mp4"
            if not video_file.exists():
                continue

            game_count += 1
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
            raise RuntimeError(
                f"未能提取任何有效样本（找到 {game_count} 场比赛但无有效标注）。"
                "请检查视频文件和标注文件的一致性。"
            )

        return samples

    def _load_clip(self, video_path: str, center_sec: float):
        """
        用 PyAV 解码指定时间段的帧，返回 [C, T, H, W] float Tensor
        """
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
            print(f"[警告] 读取视频失败 {video_path} @ {center_sec:.1f}s : {e}")

        if not frames:
            frames = [np.zeros((224, 224, 3), dtype=np.uint8)] * self.num_frames

        # 均匀采样 num_frames 帧
        indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
        clip = np.stack([frames[i] for i in indices], axis=0)  # [T, H, W, C]
        return torch.from_numpy(clip).float().permute(3, 0, 1, 2)  # [C, T, H, W]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        video_path, center_sec, label_idx = self.samples[idx]
        clip = self._load_clip(video_path, center_sec)

        sample = {"video": clip, "label": label_idx}
        if self.transform:
            sample = self.transform(sample)

        sample["label"] = torch.tensor(sample["label"], dtype=torch.long)
        return sample


# ==========================================================================
# 便利函数：模仿 Ucf101 的 API 设计
# ==========================================================================
def SoccerNet(
    data_path: str,
    clip_duration: float = 4.0,
    num_frames: int = 8,
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> SoccerNetClipDataset:
    """
    便利函数，用于创建 SoccerNet 动作识别数据集对象。

    参考 PyTorchVideo Ucf101 的设计模式，提供统一的 API。

    Args:
        data_path (str): 数据集路径
            * 若为 split 目录，如 /path/to/SN-BAS-2025/train
              则会递归查找该目录下所有 Labels-ball.json
            * 若为联赛/赛季目录，如 /path/to/train/england_efl/2019-2020
              则查找该目录及其子目录的所有 Labels-ball.json

        clip_duration (float): 以动作时间戳为中心截取的片段时长（秒），默认 4.0

        num_frames (int): 从片段中均匀采样的帧数，默认 8

        transform (Callable): 数据增强变换函数，接收 {"video": Tensor, "label": int}
                且返回相同格式。可选参数。

    Returns:
        SoccerNetClipDataset: 可直接用于 DataLoader 的数据集对象

    Example:
        >>> from torchvision.transforms import Compose
        >>> from pytorchvideo.transforms import ApplyTransformToKey, Normalize
        >>> from soccernet_dataset import SoccerNet, SoccerNetClipDataset
        >>>
        >>> # 创建简单的数据集（无变换）
        >>> dataset = SoccerNet(data_path="/path/to/SN-BAS-2025/train")
        >>>
        >>> # 创建带数据增强的数据集
        >>> transform = ApplyTransformToKey(
        ...     key="video",
        ...     transform=Compose([
        ...         Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
        ...     ]),
        ... )
        >>> dataset = SoccerNet(
        ...     data_path="/path/to/SN-BAS-2025/train",
        ...     clip_duration=4.0,
        ...     num_frames=8,
        ...     transform=transform,
        ... )
        >>>
        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(dataset, batch_size=4, num_workers=4, shuffle=True)
    """

    torch._C._log_api_usage_once("SOCCERNET.dataset.SoccerNet")

    return SoccerNetClipDataset(
        data_path=data_path,
        clip_duration=clip_duration,
        num_frames=num_frames,
        transform=transform,
    )


# ==========================================================================
# 辅助函数：获取类别信息
# ==========================================================================
def get_num_classes() -> int:
    """获取 SoccerNet 动作类别总数"""
    return NUM_CLASSES


def get_action_names() -> list:
    """获取所有动作类别名称"""
    return SOCCERNET_ACTIONS.copy()


def get_action_to_idx_map() -> dict:
    """获取动作标签到索引的映射字典"""
    return ACTION_TO_IDX.copy()


if __name__ == "__main__":
    # 演示用法
    print("=" * 60)
    print("SoccerNet Dataset Module - 演示用法")
    print("=" * 60)

    print("\n1. 动作类别信息：")
    print(f"   总类别数: {get_num_classes()}")
    print(f"   动作列表: {get_action_names()}")

    print("\n2. 创建数据集示例：")
    print("""
    from soccernet_dataset import SoccerNet
    
    dataset = SoccerNet(
        data_path="/path/to/SN-BAS-2025/train",
        clip_duration=4.0,
        num_frames=8,
        transform=your_transform,
    )
    
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, num_workers=4, shuffle=True)
    
    for batch in loader:
        videos = batch["video"]  # [B, C, T, H, W]
        labels = batch["label"]  # [B]
    """)

    print("\n3. 与原 SoccerNetClipDataset 的区别：")
    print("""
    - SoccerNet()     : 便利函数，提供简洁 API（推荐用于快速实验）
    - SoccerNetClipDataset : 直接类，提供更细粒度的控制
    
    两者核心功能完全相同，SoccerNet() 只是 SoccerNetClipDataset 的包装。
    """)
