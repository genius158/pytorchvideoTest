"""
SoccerNet 数据管理模块使用示例

展示如何使用 soccernet_dataset.py 中的 SoccerNet() 便利函数方便地加载数据。
"""

from soccernet_dataset import SoccerNet, get_num_classes, get_action_names
from torch.utils.data import DataLoader

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import Compose, Lambda, RandomCrop, RandomHorizontalFlip, CenterCrop


def example_basic():
    """例子 1：最简单的用法 - 仅加载数据，无变换"""
    print("\n[例子 1] 最基本用法：加载训练数据")
    print("-" * 60)

    # 仅需调用 SoccerNet() 便利函数
    dataset = SoccerNet(
        data_path="/path/to/SN-BAS-2025/train",
        clip_duration=4.0,
        num_frames=8,
    )

    print(f"数据集规模: {len(dataset)} 个样本")
    print(f"总类别数: {get_num_classes()}")

    # 创建 DataLoader
    loader = DataLoader(dataset, batch_size=4, num_workers=4, shuffle=True)
    print(f"批次数: {len(loader)}")

    # 遍历第一个批次
    batch = next(iter(loader))
    print(f"批次数据:")
    print(f"  - video shape: {batch['video'].shape}  # [B, C, T, H, W]")
    print(f"  - label shape: {batch['label'].shape}  # [B]")


def example_with_transform():
    """例子 2：添加数据增强变换"""
    print("\n[例子 2] 带数据增强的用法")
    print("-" * 60)

    def get_video_transform(mode="train"):
        """定义数据增强流水线"""
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]

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

    # 分别创建训练集和验证集
    train_dataset = SoccerNet(
        data_path="/path/to/SN-BAS-2025/train",
        clip_duration=4.0,
        num_frames=8,
        transform=get_video_transform("train"),
    )

    val_dataset = SoccerNet(
        data_path="/path/to/SN-BAS-2025/valid",
        clip_duration=4.0,
        num_frames=8,
        transform=get_video_transform("val"),
    )

    print(f"训练集规模: {len(train_dataset)}")
    print(f"验证集规模: {len(val_dataset)}")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        num_workers=4,
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        num_workers=4,
        shuffle=False,
    )

    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}")


def example_class_info():
    """例子 3：获取类别信息"""
    print("\n[例子 3] 查看类别信息")
    print("-" * 60)

    num_classes = get_num_classes()
    action_names = get_action_names()

    print(f"总类别数: {num_classes}")
    print(f"所有动作类别:")
    for idx, name in enumerate(action_names):
        print(f"  {idx:2d}: {name}")


def example_custom_params():
    """例子 4：自定义片段和采样参数"""
    print("\n[例子 4] 自定义参数")
    print("-" * 60)

    # 短片段（1秒），多帧采样（16帧）
    dataset_short = SoccerNet(
        data_path="/path/to/SN-BAS-2025/train",
        clip_duration=1.0,   # 只截 1 秒
        num_frames=16,       # 采样 16 帧
    )

    # 长片段（8秒），少帧采样（4帧）
    dataset_long = SoccerNet(
        data_path="/path/to/SN-BAS-2025/train",
        clip_duration=8.0,   # 截 8 秒
        num_frames=4,        # 采样 4 帧
    )

    print(f"短片段配置 (1s, 16帧)")
    print(f"  样本数: {len(dataset_short)}")

    print(f"长片段配置 (8s, 4帧)")
    print(f"  样本数: {len(dataset_long)}")

    print("\n说明:")
    print("  clip_duration 越长，捕获更多上下文")
    print("  num_frames 越多，时间分辨率越高")


if __name__ == "__main__":
    print("=" * 70)
    print("SoccerNet 数据管理模块 - 使用示例")
    print("=" * 70)

    print("\n注意：以下例子使用 /path/to/SN-BAS-2025/train 作为占位符")
    print("请替换为实际的数据集路径，例如：")
    print("  /home/nio/AI/dataset/SN-BAS-2025/train")

    # 仅演示代码结构，不实际运行（因为需要真实数据路径）
    print("\n" + "-" * 70)
    print("代码示例（不执行，请复制后修改路径）:")
    print("-" * 70)

    example_class_info()

    print("\n" + "=" * 70)
    print("关键要点:")
    print("=" * 70)
    print("""
1. SoccerNet() 是参考 PyTorchVideo.Ucf101() 设计的便利函数
   - 返回 SoccerNetClipDataset 对象
   - 可直接用于 DataLoader

2. 核心参数:
   - data_path: 数据集路径（递归查找）
   - clip_duration: 片段时长（秒）
   - num_frames: 采样帧数
   - transform: 数据增强变换

3. 返回值格式:
   batch["video"]  : [B, C, T, H, W]  # 视频片段张量
   batch["label"]  : [B]              # 动作标签

4. 相比直接使用 SoccerNetClipDataset 的好处:
   - API 更简洁，更符合 PyTorchVideo 规范
   - 易于快速原型设计和实验
   - 代码清晰，易读易维护
    """)
