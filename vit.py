import torch
import torch.nn as nn
import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import re
import numpy as np
from tqdm import tqdm  # 修正拼写错误
import matplotlib.pyplot as plt
import csv

# 全局设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class ViTFeatureExtractor(nn.Module):
    def __init__(self):
        super(ViTFeatureExtractor, self).__init__()
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        
        # 冻结所有参数
        for param in self.vit.parameters():
            param.requires_grad_(False)
            
    def forward(self, x):
        x = self.vit._process_input(x)
        batch_class_token = self.vit.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = x + self.vit.encoder.pos_embedding
        x = self.vit.encoder(x)
        return x[:, 0]  # 返回类别token特征

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTFeatureExtractor().to(device).eval()

def exponential_similarity(distance, gamma=0.05):
    """指数衰减相似度计算"""
    return np.exp(-gamma * distance)

def process_group(group_dir):
    """改进后的数据处理函数，返回特征和图像ID"""
    try:
        stage_files = {}
        group_name = os.path.basename(group_dir)
        image_ids = []
        
        # 收集文件并生成ID
        for fname in os.listdir(group_dir):
            match = re.search(r"_0([1-5])\.png$", fname)
            if match:
                stage = int(match.group(1))
                if stage in stage_files:
                    print(f"重复阶段文件: {fname}")
                    return None
                stage_files[stage] = os.path.join(group_dir, fname)
                image_ids.append(f"{group_name}_0{stage}")  # 生成完整ID
        
        if set(stage_files.keys()) != {1,2,3,4,5}:
            return None
            
        # 保持图像加载顺序
        img_tensors = []
        for stage in sorted(stage_files.keys()):
            img = Image.open(stage_files[stage]).convert('RGB')
            img_tensors.append(preprocess(img))
        
        # 批量处理
        batch = torch.stack(img_tensors).to(device)
        with torch.no_grad():
            features = model(batch).cpu().numpy()
            
        return image_ids, features
    except Exception as e:
        print(f"处理异常: {str(e)}")
        return None

def main():
    base_dir = r"C:\Users\fairy刘\Desktop\res\res\datasets\Image"
    
    groups = [os.path.join(base_dir, d) for d in os.listdir(base_dir)
             if os.path.isdir(os.path.join(base_dir, d))]
    groups.sort(key=lambda x: int(re.search(r"\d+", os.path.basename(x)).group()))
    
    all_features = []
    similarity_records = []
    all_similarities = []
    
    for group_dir in tqdm(groups, desc="处理数据组"):
        result = process_group(group_dir)
        if result is None:
            continue
        image_ids, features = result
        
        if features.shape[0] != 5:
            continue
            
        # 存储特征数据
        for img_id, feat in zip(image_ids, features):
            all_features.append((img_id, feat))
        
        # 计算相似度
        similarities = []
        for i in range(4):
            img1, img2 = image_ids[i], image_ids[i+1]
            dist = np.linalg.norm(features[i] - features[i+1])
            similarity = exponential_similarity(dist)
            similarities.append(similarity)
            similarity_records.append((img1, img2, similarity))
        
        all_similarities.append(similarities)
    
    # 保存特征数据
    with open("image_features_vit.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_id"] + [f"feature_{i}" for i in range(768)])
        for img_id, features in all_features:
            writer.writerow([img_id] + features.tolist())
    
    # 保存相似度记录
    with open("similarity_scores_vit.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image1", "image2", "similarity"])
        writer.writerows(similarity_records)
    
    # 统计分析和可视化
    all_sims = np.array(all_similarities)
    avg_sims = np.mean(all_sims, axis=0)
    std_sims = np.std(all_sims, axis=0)
    
    print("\n相邻阶段平均相似度（Vision Transformer）：")
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
    plt.title("甲骨文演化阶段相似度趋势（Vision Transformer）", fontsize=14, pad=20)
    
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
    plt.savefig('vit_similarity.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()