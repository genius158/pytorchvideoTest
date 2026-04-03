import torch
import torch.nn as nn
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample
from torchvision.transforms import Compose, Lambda, CenterCrop
from pytorchvideo.data.encoded_video import EncodedVideo

# 1. 必须定义与训练时完全一致的模型结构
def load_my_model(ckpt_path, num_classes=20):
    model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_xs', pretrained=False)
    # 替换分类头（必须和训练时一模一样）
    in_features = model.blocks[5].proj.in_features
    model.blocks[5].proj = nn.Linear(in_features, num_classes)
    
    # 加载权重
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    # 自动处理 Lightning 的前缀问题
    state_dict = {k.replace("model.", ""): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model

# 2. 极简预处理
transform = Compose([
    Lambda(lambda x: x / 255.0),
    UniformTemporalSubsample(4), # 采样8帧
    CenterCrop(160),             # 裁切大小
])

# 3. 开始预测
CKPT = ".checkpoints/last.ckpt"   # 换成你的文件名
VIDEO = "video/BrushingTeeth/v_BrushingTeeth_g01_c01.avi"  # 换成你想测试的视频

device = "cpu"
model = load_my_model(CKPT)
video = EncodedVideo.from_path(VIDEO)
video_data = video.get_clip(0, 1.0) # 取前2秒

# [Batch, Channel, Time, Height, Width]
inputs = transform(video_data["video"]).unsqueeze(0) 

with torch.no_grad():
    preds = model(inputs)
    pred_class = preds.argmax(dim=1).item()
    print(f"预测类别索引: {pred_class}")