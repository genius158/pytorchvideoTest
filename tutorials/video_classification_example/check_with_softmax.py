import torch
from trainx3d_v3 import VideoClassificationLightningModule

# 加载模型
ckpt = VideoClassificationLightningModule.load_from_checkpoint(
    ".checkpoints/last.ckpt", 
    map_location="cpu"
)
model = ckpt.model
model.eval()

# 测试输出
with torch.no_grad():
    x = torch.rand(1, 3, 8, 160, 160)
    output = model(x)
    
    print(f"输出形状：{output.shape}")
    print(f"输出范围：[{output.min():.4f}, {output.max():.4f}]")
    print(f"输出和：{output.sum(dim=1)}")
    print(f"Softmax 后和：{output.softmax(dim=1).sum(dim=1)}")
    
    # 判断是否为概率分布
    if torch.allclose(output.sum(dim=1), torch.ones(1), atol=1e-4):
        print("✅ 模型输出已是概率分布（包含 Softmax）")
    else:
        print("❌ 模型输出是 logits（不包含 Softmax）")