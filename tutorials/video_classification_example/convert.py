import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

from train_soccernet_v4 import SoccerNetClassifier


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def replace_swish_with_silu(model: nn.Module) -> int:
    """
    递归遍历模型，将无法导出的 Swish 激活层替换为官方支持的 nn.SiLU。
    返回替换数量，便于确认修复是否生效。
    """
    replaced = 0
    for name, module in model.named_children():
        class_name = module.__class__.__name__.lower()
        if "swish" in class_name:
            setattr(model, name, nn.SiLU())
            replaced += 1
            continue
        replaced += replace_swish_with_silu(module)
    return replaced


def export_for_android(
    ckpt_path: str,
    output_name: str = "x3d_model_android",
    input_shape=(1, 3, 5, 224, 224),
) -> None:
    print("--- 开始转换过程 ---")
    device = resolve_device()
    print(f"当前导出设备: {device}")

    # 1. 加载 Checkpoint
    try:
        checkpoint = SoccerNetClassifier.load_from_checkpoint(
            ckpt_path,
            map_location=device,
        )
        model = checkpoint.model.to(device).eval()

        # 2. 核心修复步骤：替换不支持导出的 Swish 函数
        print("正在修复 Swish 层以兼容 TorchScript...")
        replaced_count = replace_swish_with_silu(model)
        print(f"1. 成功加载模型，已替换 Swish 层数量: {replaced_count}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    # 3. 准备 dummy 输入 [N, C, T, H, W]
    try:
        dummy_input = torch.rand(*input_shape, device=device)
    except Exception as e:
        print(f"dummy 输入 shape 非法: {input_shape}, error: {e}")
        return

    # 4. 先 script，失败后回退 trace（script 通常更稳健）
    print("2. 正在进行 TorchScript 导出...")
    try:
        with torch.inference_mode():
            try:
                scripted_model = torch.jit.script(model)
                print("   - 使用 script 导出成功")
            except Exception as script_error:
                print(f"   - script 失败，回退 trace: {script_error}")
                scripted_model = torch.jit.trace(model, dummy_input, check_trace=False)
                print("   - 使用 trace 导出成功")

        # 移动端优化前统一转到 CPU
        scripted_model = scripted_model.to("cpu")

        # 5. 为 Android 进行优化
        print("3. 正在进行移动端算子优化...")
        optimized_model = optimize_for_mobile(scripted_model)

        # 6. 保存
        output_ptl = Path(f"{output_name}.ptl")
        output_ptl.parent.mkdir(parents=True, exist_ok=True)
        optimized_model._save_for_lite_interpreter(str(output_ptl))
        print(f"✅ 转换成功！Android 文件: {output_ptl}")

    except Exception as e:
        # GPU 导出失败时自动回退 CPU 重试
        if device.type == "cuda":
            print(f"❌ GPU 导出失败，自动回退 CPU 重试: {e}")
            try:
                cpu_device = torch.device("cpu")
                model = model.to(cpu_device).eval()
                dummy_input = torch.rand(*input_shape, device=cpu_device)
                with torch.inference_mode():
                    try:
                        scripted_model = torch.jit.script(model)
                        print("   - CPU script 导出成功")
                    except Exception as script_error:
                        print(f"   - CPU script 失败，回退 trace: {script_error}")
                        scripted_model = torch.jit.trace(model, dummy_input, check_trace=False)
                        print("   - CPU trace 导出成功")

                print("3. 正在进行移动端算子优化...")
                optimized_model = optimize_for_mobile(scripted_model)
                output_ptl = Path(f"{output_name}.ptl")
                output_ptl.parent.mkdir(parents=True, exist_ok=True)
                optimized_model._save_for_lite_interpreter(str(output_ptl))
                print(f"✅ 转换成功！Android 文件: {output_ptl}")
                return
            except Exception as retry_e:
                print(f"❌ CPU 回退导出失败: {retry_e}")
                return

        print(f"❌ TorchScript 导出失败: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Export X3D checkpoint to Android .ptl")
    parser.add_argument("--ckpt", default=".checkpoints_soccernet_v7/last.ckpt", help="Checkpoint path")
    parser.add_argument("--out", default="x3d_model_android", help="Output file prefix")
    parser.add_argument("--shape", nargs=5, type=int, default=[1, 3, 5, 224, 224], help="Input shape: N C T H W")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_for_android(args.ckpt, args.out, tuple(args.shape))