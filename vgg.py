import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import re
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# 全局设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        original_vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        
        # 原始特征提取层保持不变
        self.features = original_vgg.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        # 新增降维适配层（唯一修改点）
        self.adaptor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512),  # 25088 → 512
            nn.ReLU(inplace=True),
            nn.LayerNorm(512)
        )
        
        # 冻结所有参数（包括新增适配层）
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return self.adaptor(x)  # 输出维度512

# 保持原始预处理流程
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGG16FeatureExtractor().to(device).eval()

def exponential_similarity(distance, gamma=0.5):  # 唯一参数调整
    """指数衰减相似度计算"""
    return np.exp(-gamma * distance)

def process_group(group_dir):
    """仅添加特征归一化"""
    try:
        # 原始文件处理逻辑
        stage_files = {}
        for fname in os.listdir(group_dir):
            match = re.search(r"_0([1-5])\.png$", fname)
            if match:
                stage = int(match.group(1))
                if stage in stage_files:
                    print(f"重复阶段文件: {fname}")
                    return None
                stage_files[stage] = os.path.join(group_dir, fname)
        
        if set(stage_files.keys()) != {1,2,3,4,5}:
            return None
            
        # 原始图像加载逻辑
        img_tensors = []
        for stage in sorted(stage_files.keys()):
            img = Image.open(stage_files[stage]).convert('RGB')
            img_tensors.append(preprocess(img))
        
        # 添加特征归一化（唯一新增代码）
        batch = torch.stack(img_tensors).to(device)
        with torch.no_grad():
            features = model(batch).cpu().numpy()
            features = features / np.linalg.norm(features, axis=1, keepdims=True)  # L2归一化
            
        return features
    except Exception as e:
        print(f"处理异常: {str(e)}")
        return None

# 以下所有代码保持原样
def main():
    base_dir = r"C:\Users\fairy刘\Desktop\res\data\Image"
    
    groups = [os.path.join(base_dir, d) for d in os.listdir(base_dir)
             if os.path.isdir(os.path.join(base_dir, d))]
    groups.sort(key=lambda x: int(re.search(r"\d+", os.path.basename(x)).group()))
    
    all_similarities = []
    
    for group_dir in tqdm(groups, desc="处理数据组"):
        features = process_group(group_dir)
        if features is not None and features.shape[0] == 5:
            similarities = []
            for i in range(4):
                dist = np.linalg.norm(features[i] - features[i+1])
                similarities.append(exponential_similarity(dist))
            all_similarities.append(similarities)
    
    all_sims = np.array(all_similarities)
    avg_sims = np.mean(all_sims, axis=0)
    std_sims = np.std(all_sims, axis=0)
    
    print("\n相邻阶段平均相似度（优化VGG16）：")
    transitions = ["obi→bi", "bi→ss", "ss→os", "os→rs"]
    for trans, avg, std in zip(transitions, avg_sims, std_sims):
        print(f"{trans}: {avg:.2%} ± {std:.2%}")

    plt.figure(figsize=(10, 6), facecolor='white')
    x = np.arange(len(transitions))
    
    main_line, = plt.plot(x, avg_sims, 
                         color='#E63946', 
                         marker='o', 
                         markersize=8,
                         linewidth=2,
                         label='平均相似度')
    
    std_area = plt.fill_between(x, 
                               avg_sims - std_sims,
                               avg_sims + std_sims,
                               color='#F4A0A8',
                               alpha=0.3,
                               label='标准差范围')
    
    plt.xticks(x, transitions)
    plt.ylim(0, 1.0)
    plt.yticks(np.linspace(0, 1.0, 6), 
              ['0%', '20%', '40%', '60%', '80%', '100%'])
    plt.xlabel("阶段过渡", fontsize=12)
    plt.ylabel("相似度", fontsize=12)
    plt.title("甲骨文演化阶段相似度趋势（优化VGG16）", fontsize=14, pad=20)
    
    for xi, yi in zip(x, avg_sims):
        plt.text(xi, yi+0.03, f"{yi:.1%}",
                ha='center',
                va='bottom',
                fontsize=10,
                color='#E63946')
    
    plt.legend(handles=[main_line, std_area],
              labels=['平均相似度', '标准差范围'],
              loc='upper right',
              frameon=False)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.gca().set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig('optimized_vgg16.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()