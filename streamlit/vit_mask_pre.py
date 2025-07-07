import timm
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms
import torch
from sklearn.cluster import DBSCAN  # 用于文本框合并


class ViTEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.vit = timm.create_model("vit_base_patch16_224", pretrained=pretrained, num_classes=0)
        self.patch_size = self.vit.patch_embed.patch_size[0]
        
    def forward(self, x):
        # x: [B, 3, H, W]
        features = self.vit.forward_features(x)  # [B, num_patches + 1, dim]
        cls_token, patch_tokens = features[:, 0], features[:, 1:]
        
        # 将序列还原为2D特征图（假设输入为224x224，patch=16）
        B, num_patches, dim = patch_tokens.shape
        H = W = int(num_patches ** 0.5)
        patch_tokens = patch_tokens.permute(0, 2, 1).reshape(B, dim, H, W)
        return patch_tokens  # [B, dim, H_patch, W_patch]


class Decoder(nn.Module):
    def __init__(self, in_dim, out_channels=1):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(768, 256, kernel_size=4, stride=4),  # [B, 256, 56, 56]
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=4),   # [B, 64, 224, 224]
            nn.Conv2d(64, 1, kernel_size=1),                       # [B, 1, 224, 224]
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.up(x)  # [B, 1, H, W]


class ViTForMaskPrediction(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ViTEncoder()
        self.decoder = Decoder(in_dim=self.encoder.vit.embed_dim)
        
    def forward(self, x):
        features = self.encoder(x)  # [B, dim, H_patch, W_patch]
        mask = self.decoder(features)  # [B, 1, H, W]
        return mask
    

def enhance_heatmap(heatmap):
    """增强浅色区域的对比度"""
    # 方法1：非线性增强（突出低概率区域）
    enhanced = np.power(heatmap, 0.5)  # 伽马校正
    
    # 方法2：自适应阈值（针对局部浅色区域）
    binary = cv2.adaptiveThreshold(
        (enhanced*255).astype(np.uint8), 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        blockSize=11, 
        C=2
    )
    return binary


def extract_text_boxes(heatmap, original_img_shape, min_area=0, edge_margin_ratio=0.05, angle_threshold=15):
    """
    从热力图中提取并筛选文本框
    
    参数:
        heatmap: 模型输出的热力图 (0-1 float32, 尺寸可能小于原图)
        original_img_shape: 原图尺寸 (H, W)
        min_area: 最小文本框面积阈值 (像素)
        edge_margin_ratio: 边缘区域比例 (靠近此比例的边缘小框会被过滤)
        angle_threshold: 角度阈值 (度)，小于此值视为平行文本框
    返回:
        List[tuple]: 每个文本框为 (x1, y1, x2, y2) 坐标
    """
    # 1. 热力图二值化并缩放到原图尺寸
    _, binary = cv2.threshold((heatmap * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.resize(binary, (original_img_shape[1], original_img_shape[0]))
    binary = enhance_heatmap(binary)
    
    # 2. 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 3. 初步筛选轮廓
    boxes = []
    h, w = original_img_shape
    edge_margin = int(min(h, w) * edge_margin_ratio)

    for cnt in contours:
        # 获取旋转矩形 (带角度)
        rect = cv2.minAreaRect(cnt)
        if not rect:
            # 多边形近似（精度可调）
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            # 跳过简单矩形（顶点数<=4）
            if len(approx) <= 4:
                continue
                
            # 获取最小外接矩形（作为备用框）
            rect = cv2.boundingRect(approx)
        
        box = cv2.boxPoints(rect).astype(np.int32)
        # # 计算矩形参数
        (x, y), (width, height), angle = rect
        area = width * height
        if area < min_area:
            continue
        if area > binary.shape[0] * binary.shape[1] * 0.8:
            continue
        # 条件2: 边缘小框过滤 (水印噪声)
        if (min(x, w - x) < edge_margin or min(y, h - y) < edge_margin) and max(width, height) < 0.1 * min(h, w):
            continue
        boxes.append((box, angle))
    
    # 4. 合并近似平行的文本框
    merged_boxes = merge_parallel_boxes(boxes, angle_threshold)

    # 转换为 [x1,y1,x2,y2] 格式
    final_boxes = []
    for box in merged_boxes:
        x1, y1 = np.min(box, axis=0)
        x2, y2 = np.max(box, axis=0)
        final_boxes.append((x1, y1, x2, y2))
    
    # 5. 合并重叠框
    final_boxes = merge_nested_boxes(final_boxes)

    # 5. 若无检测框，添加默认框
    if len(final_boxes) == 0:
        default_width = int(w * 0.6)  # 默认宽度为图像宽度的60%
        default_height = int(h * 0.1)  # 默认高度为图像高度的10%
        x1 = int((w - default_width) / 2)  # 水平居中
        y1 = h - int(default_height * 1.5)  # 位于底部上方1.5倍高度处
        x2 = x1 + default_width
        y2 = y1 + default_height
        final_boxes.append((x1, y1, x2, y2))
    
    return final_boxes


def merge_nested_boxes(boxes):
    if len(boxes) <= 1:
        return boxes
    
    # 按面积从大到小排序（优先处理大框）
    sorted_boxes = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    
    merged_boxes = []
    for i, box in enumerate(sorted_boxes):
        x1, y1, x2, y2 = box
        is_nested = False
        
        # 检查是否被已合并的大框包含
        for merged in merged_boxes:
            mx1, my1, mx2, my2 = merged
            if x1 >= mx1 and y1 >= my1 and x2 <= mx2 and y2 <= my2:
                is_nested = True
                break
        
        # 如果不是嵌套框则保留
        if not is_nested:
            merged_boxes.append(box)
    
    return merged_boxes


def merge_parallel_boxes(boxes, angle_threshold=15):
    """
    合并角度接近的文本框 (DBSCAN聚类)
    """
    if len(boxes) == 0:
        return []
    
    # 提取角度和中心点
    angles = np.array([angle for _, angle in boxes])
    centers = np.array([np.mean(box, axis=0) for box, _ in boxes])
    
    # 角度归一化到 [-90, 90]
    angles_norm = np.where(angles > 90, angles - 180, angles)
    
    # 使用DBSCAN聚类 (角度+空间位置)
    features = np.column_stack([
        centers[:, 0] / 1000,  # 归一化
        centers[:, 1] / 1000,
        angles_norm / 90       # 角度归一化
    ])
    
    clustering = DBSCAN(eps=0.3, min_samples=1).fit(features)
    labels = clustering.labels_
    
    # 合并同簇的box
    merged = []
    for label in set(labels):
        cluster_boxes = [box for box, l in zip(boxes, labels) if l == label]
        
        # 取所有点的最小外接矩形
        all_points = np.vstack([box for box, _ in cluster_boxes])
        rect = cv2.minAreaRect(all_points)
        merged_box = cv2.boxPoints(rect)
        merged.append(merged_box.astype(np.int32))
    
    return merged

def preprocess_input(image_path, img_size=(224, 224)):
    # 1. 读取图像
    image = cv2.imread(image_path)  # [H, W, 3]
    if image is None: print('image error', image_path)

    # 2. 记录原本的尺寸
    original_shape = image.shape[:2]

    # 3. 调整尺寸
    image = cv2.resize(image, img_size)

    # 4. 图像归一化（Imagenet标准）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image)  # [3, H, W]

    return image, original_shape


# 加载模型
def getModel(language='ch'):
    CH_MODEL_PATH = '..\model\ch_model_best2.pth'
    EN_MODEL_PATH = '..\model\en_model_best2.pth'

    if language == 'ch':
        model = torch.load(CH_MODEL_PATH, weights_only=False)
    else:
        model = torch.load(EN_MODEL_PATH, weights_only=False)
    return model

# 从图获得box
def GetBoxesByPath(image_path, model=None):
    if model is None:
        model = getModel()
    model.eval()
    image, original_shape = preprocess_input(image_path)
    image = image.unsqueeze(0).cuda()
    # 假设模型输出热力图 (0-1 float32)
    heatmap = model(image)[0, 0].cpu().detach().numpy()   # 形状 [H, W]
    # 提取文本框
    boxes = extract_text_boxes(heatmap, original_shape)
    return boxes


# 使用示例
if __name__ == '__main__':
    # from vit_mask_pre import *

    # 加载模型
    model = getModel(language='ch')
    # 获取image_path
    image_path = f'../image/chinese_meme_notext/test0001.png'
    # 得到boxe，每个文本框为 (x1, y1, x2, y2) 坐标
    boxes = GetBoxesByPath(image_path, model)