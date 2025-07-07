import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import re
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        original_vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = original_vgg.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.adaptor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.LayerNorm(512)
        )
        for param in self.parameters():
            param.requires_grad_(False)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return self.adaptor(x)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGG16FeatureExtractor().to(device).eval()

def exponential_similarity(distance, gamma=0.5):
    return np.exp(-gamma * distance)

def process_group(group_dir):
    try:
        stage_files = {}
        for fname in os.listdir(group_dir):
            if match := re.search(r"_0([1-5])\.png$", fname):
                if (stage := int(match.group(1))) in stage_files:
                    print(f"重复阶段文件: {fname}")
                    return None
                stage_files[stage] = os.path.join(group_dir, fname)
        
        if set(stage_files.keys()) != {1,2,3,4,5}:
            print(f"目录 {os.path.basename(group_dir)} 缺少阶段文件")
            return None
            
        img_tensors = []
        for stage in sorted(stage_files.keys()):
            with Image.open(stage_files[stage]).convert('RGB') as img:
                img_tensors.append(preprocess(img))
        
        batch = torch.stack(img_tensors).to(device)
        with torch.no_grad():
            features = model(batch).cpu().numpy()
            return features / np.linalg.norm(features, axis=1, keepdims=True)
    except Exception as e:
        print(f"处理失败 [{os.path.basename(group_dir)}]: {str(e)}")
        return None

def main():
    # 交互式输入处理
    input_paths = input("请输入多个目录路径（用分号分隔）:\n").strip()
    group_dirs = [p.strip() for p in input_paths.split(';') if p.strip()]
    
    all_similarities = []
    success_count = 0
    
    for path in group_dirs:
        if not os.path.exists(path):
            print(f"× 路径不存在: {path}")
            continue
            
        if features := process_group(path):
            similarities = [
                exponential_similarity(np.linalg.norm(features[i] - features[i+1]))
                for i in range(4)
            ]
            all_similarities.append(similarities)
            success_count += 1
            print(f"√ 成功处理: {os.path.basename(path)}")
        else:
            print(f"× 处理失败: {os.path.basename(path)}")
    
    if not all_similarities:
        print("错误：没有有效数据可供分析")
        return
    
    # 统计计算
    sim_matrix = np.array(all_similarities)
    avg_sims = np.mean(sim_matrix, axis=0)
    std_sims = np.std(sim_matrix, axis=0)
    conf_interval = 1.96 * std_sims / np.sqrt(len(sim_matrix))  # 95%置信区间
    
    # 结果展示
    transitions = ["OBI→BI", "BI→SS", "SS→OS", "OS→RS"]
    print("\n分析结果：")
    print(f"有效数据组数: {success_count}")
    print("过渡阶段 | 平均相似度 | 标准差 | 置信区间(95%)")
    for t, avg, std, ci in zip(transitions, avg_sims, std_sims, conf_interval):
        print(f"{t}\t{avg:.2%} ± {std:.2%}\t({avg-ci:.2%} - {avg+ci:.2%})")
    
    # 可视化增强
    plt.figure(figsize=(12, 7), facecolor='#F5F5F5')
    x = np.arange(len(transitions))
    
    # 主趋势线
    main_line = plt.plot(x, avg_sims, 
                        color='#2A5B82',
                        marker='D',
                        markersize=10,
                        linewidth=3,
                        label='平均相似度',
                        zorder=3)
    
    # 置信区间
    plt.errorbar(x, avg_sims, yerr=conf_interval,
                fmt='none', ecolor='#E74C3C', 
                elinewidth=2, capsize=15,
                capthick=2, zorder=2)
    
    # 数据点分布
    for i, col in enumerate(sim_matrix.T):
        plt.scatter([i]*len(col), col, 
                   color='#27AE60', alpha=0.4,
                   s=80, edgecolor='white',
                   label='单组数据' if i == 0 else "")
    
    # 样式设置
    plt.xticks(x, transitions, fontsize=12)
    plt.yticks(np.arange(0, 1.1, 0.2), 
              [f"{i:.0%}" for i in np.arange(0, 1.1, 0.2)],
              fontsize=11)
    plt.ylim(-0.05, 1.15)
    
    plt.xlabel("演化阶段过渡", fontsize=13, labelpad=12)
    plt.ylabel("特征相似度", fontsize=13, labelpad=12)
    plt.title(f"甲骨文演化阶段相似度分析（{success_count}组数据平均）\n", 
             fontsize=15, pad=25)
    
    # 图例重排
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=[main_line[0], handles[-1], handles[1]],
              labels=['平均相似度', '单组数据', '95%置信区间'],
              loc='upper center',
              bbox_to_anchor=(0.5, -0.12),
              ncol=3,
              frameon=False,
              fontsize=11)
    
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    # 保存多种格式
    plt.savefig('multi_group_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('multi_group_analysis.svg')  # 矢量格式
    plt.show()

if __name__ == "__main__":
    main()