import os
import re
import csv
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# 全局设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class StageFeatureAnalyzer(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])
        for param in self.feature_extractor.parameters():
            param.requires_grad_(False)
            
    def forward(self, x):
        return self.feature_extractor(x).squeeze()

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = StageFeatureAnalyzer().to(device).eval()

def parse_stage_number(fname):
    """直接提取_后的阶段编号（01-05）"""
    match = re.search(r"_(\d{2})\.png$", fname)
    if match and match.group(1) in ["01","02","03","04","05"]:
        return int(match.group(1))
    return None

def process_group(group_dir):
    """新版组处理逻辑"""
    try:
        # 收集所有阶段文件
        stage_files = {}
        for fname in os.listdir(group_dir):
            if not fname.lower().endswith(".png"):
                continue
                
            stage = parse_stage_number(fname)
            if stage is None:
                continue
                
            if stage in stage_files:
                print(f"重复阶段文件: {fname} 与 {stage_files[stage]}")
                return None
                
            stage_files[stage] = os.path.join(group_dir, fname)
        
        if set(stage_files.keys()) != {1,2,3,4,5}:
            print(f"实际文件列表: {os.listdir(group_dir)}")
            return None
            
        # 按阶段顺序加载
        img_tensors = []
        for stage in sorted(stage_files.keys()):
            img = Image.open(stage_files[stage]).convert('RGB')
            img_tensors.append(preprocess(img))
        
        batch = torch.stack(img_tensors).to(device)
        with torch.no_grad():
            features = model(batch).cpu().numpy()
            features = features / np.linalg.norm(features, axis=1, keepdims=True)
        
        group_name = os.path.basename(group_dir)
        image_ids = [f"{group_name}_{stage:02d}" for stage in sorted(stage_files.keys())]
        
        return list(zip(image_ids, features))
    except Exception as e:
        print(f"处理失败: {os.path.basename(group_dir)} - {str(e)}")
        return None

def exponential_similarity(distance, gamma=0.05):
    """指数衰减相似度计算"""
    return np.exp(-gamma * distance)

def main():
    base_dir = r"C:\Users\fairy刘\Desktop\res\res\datasets\Image"
    
    # 获取所有组目录
    groups = [os.path.join(base_dir, d) for d in os.listdir(base_dir)
             if os.path.isdir(os.path.join(base_dir, d))]
    
    # 存储结果
    all_features = []
    similarity_records = []
    all_similarities = []
    
    # 处理数据
    for group_dir in tqdm(groups, desc="处理数据组"):
        features_info = process_group(group_dir)
        if features_info is not None and len(features_info) == 5:
            all_features.extend(features_info)
            
            image_ids = [item[0] for item in features_info]
            feature_vectors = [item[1] for item in features_info]
            
            similarities = []
            for i in range(4):
                img1 = image_ids[i]
                img2 = image_ids[i+1]
                dist = np.linalg.norm(feature_vectors[i] - feature_vectors[i+1])
                similarity = exponential_similarity(dist)
                similarities.append(similarity)
                similarity_records.append((img1, img2, similarity))
            
            all_similarities.append(similarities)
    
    # 保存特征文件
    with open("image_features.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_id"] + [f"feature_{i}" for i in range(len(all_features[0][1]))])
        for image_id, feature in all_features:
            writer.writerow([image_id] + feature.tolist())
    
    # 保存相似度文件
    with open("similarity_scores.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image1", "image2", "similarity"])
        writer.writerows(similarity_records)
    
    # 统计分析
    all_sims = np.array(all_similarities)
    avg_sims = np.mean(all_sims, axis=0)
    std_sims = np.std(all_sims, axis=0)
    
    print("\n相邻阶段平均相似度（指数衰减法）：")
    transitions = ["1→2", "2→3", "3→4", "4→5"]
    for trans, avg, std in zip(transitions, avg_sims, std_sims):
        print(f"{trans}: {avg:.2%} ± {std:.2%}")

    # 可视化
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
    plt.title("甲骨文演化阶段相似度趋势（指数衰减法）", fontsize=14, pad=20)
    
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