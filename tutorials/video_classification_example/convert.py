import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile
from trainx3d_v3 import VideoClassificationLightningModule

# 关键：定义一个与 TorchScript 兼容的 SiLU 层来替换原来的 Swish
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def replace_swish_with_silu(model):
    """
    递归遍历模型，将所有无法导出的 Swish 层替换为官方支持的 SiLU
    """
    for name, module in model.named_children():
        if 'Swish' in module.__class__.__name__:
            setattr(model, name, nn.SiLU())
        else:
            replace_swish_with_silu(module)

def export_for_android(ckpt_path, output_name="x3d_model_android"):
    print(f"--- 开始转换过程 ---")
    
    # 1. 加载 Checkpoint
    try:
        checkpoint = VideoClassificationLightningModule.load_from_checkpoint(
            ckpt_path, 
            map_location="cpu"
        )
        model = checkpoint.model
        
        # 2. 核心修复步骤：替换不支持导出的 Swish 函数
        print("正在修复 Swish 层以兼容 TorchScript...")
        replace_swish_with_silu(model)
        
        model.eval()
        print("1. 成功加载并修复模型结构")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    # 3. 准备 dummy 输入 [1, 3, 8, 160, 160]
    dummy_input = torch.rand(1, 3, 8, 160, 160)

    # 4. 执行 TorchScript Tracing
    print("2. 正在进行 TorchScript Tracing (现在应该可以成功了)...")
    try:
        # 使用 check_trace=False 减少不必要的警告
        traced_model = torch.jit.trace(model, dummy_input, check_trace=False)
        
        # 5. 为 Android 进行优化
        print("3. 正在进行移动端算子优化...")
        optimized_model = optimize_for_mobile(traced_model)

        # 6. 保存
        output_ptl = f"{output_name}.ptl"
        optimized_model._save_for_lite_interpreter(output_ptl)
        print(f"✅ 转换成功！Android 文件: {output_ptl}")
        
    except Exception as e:
        print(f"❌ Tracing 失败: {e}")

if __name__ == "__main__":
    # 确保此文件名正确
    MY_CKPT = ".checkpoints/last.ckpt" 
    export_for_android(MY_CKPT)