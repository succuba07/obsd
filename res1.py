import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import os
import re
import numpy as np
from tqdm import tqdm
import csv

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])
        
        for param in self.feature_extractor.parameters():
            param.requires_grad_(False)
            
    def forward(self, x):
        return self.feature_extractor(x).squeeze()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork().to(device).eval()

def process_group(group_dir):
    try:
        stage_files = {}
        for fname in os.listdir(group_dir):
            match = re.search(r"_0([1-5])\.png$", fname)
            if match:
                stage = int(match.group(1))
                stage_files[stage] = os.path.join(group_dir, fname)
        
        if set(stage_files.keys()) != {1,2,3,4,5}:
            print(f"阶段不完整: {group_dir}")
            return None
            
        img_tensors = []
        for stage in sorted(stage_files.keys()):
            try:
                img = Image.open(stage_files[stage]).convert('RGB')
                img_tensors.append(preprocess(img))
            except Exception as e:
                print(f"文件损坏: {stage_files[stage]} - {str(e)}")
                return None
        
        batch = torch.stack(img_tensors).to(device)
        with torch.no_grad():
            features = model(batch).cpu().numpy()
            if features.ndim != 2 or features.shape[1] != 2048:
                print(f"特征维度异常: {features.shape}")
                return None
        
        group_name = os.path.basename(group_dir)
        image_ids = [f"{group_name}_0{stage}" for stage in sorted(stage_files.keys())]
        return list(zip(image_ids, features))
    except Exception as e:
        print(f"处理失败: {group_dir} - {str(e)}")
        return None

def main():
    base_dir = r"C:\Users\fairy刘\Desktop\res\datasets\Image"
    
    groups = [os.path.join(base_dir, d) for d in os.listdir(base_dir)
             if os.path.isdir(os.path.join(base_dir, d))]
    groups.sort()
    
    all_features = []
    similarity_records = []
    
    for group_dir in tqdm(groups, desc="处理数据组"):
        features_info = process_group(group_dir)
        if features_info is None:
            continue
            
        all_features.extend(features_info)
        
        # 记录处理进度
        print(f"成功处理: {os.path.basename(group_dir)} 包含{len(features_info)}条特征")
        
        # 保存实时数据
        with open("resnet_features_full.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for image_id, feat in features_info:
                writer.writerow([image_id] + feat.tolist())
        
        # 实时写入相似度
        with open("similarity_full.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            image_ids = [item[0] for item in features_info]
            features = [item[1] for item in features_info]
            for i in range(4):
                dist = np.linalg.norm(features[i] - features[i+1])
                similarity = np.exp(-0.05 * dist)
                writer.writerow([image_ids[i], image_ids[i+1], similarity])
    
    # 生成最终文件头
    with open("resnet_features_full.csv", "r+") as f:
        content = f.read()
        f.seek(0)
        f.write("image_id," + ",".join(f"feature_{i}" for i in range(2048)) + "\n" + content)
    
    with open("similarity_full.csv", "r+") as f:
        content = f.read()
        f.seek(0)
        f.write("image1,image2,similarity\n" + content)

if __name__ == "__main__":
    main()