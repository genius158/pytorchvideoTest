import torch
from pytorchvideo.models import create_resnet

# 创建一个预训练的 ResNet 模型
model = create_resnet(
    input_channel=3,
    model_depth=50,
    model_num_class=400,
    stem_dim_out=64,
    stem_conv_kernel_size=(3, 7, 7),
    stem_conv_stride=(1, 2, 2),
    pool_size=(1, 3, 3),
    head_pool_kernel_size=(8, 7, 7),
    head_activation="softmax",
    head_output_with_global_average=True,
)

# 加载预训练权重
model.load_state_dict(torch.load("path_to_pretrained_weights.pth"))

# 设置模型为评估模式
model.eval()

# 进行推理
with torch.no_grad():
    input_video = torch.randn(1, 3, 8, 224, 224)  # 示例输入
    output = model(input_video)
    print(output)