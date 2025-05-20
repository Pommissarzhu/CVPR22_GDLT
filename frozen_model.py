# vst_mlp.py
# 端到端 FIS-V 视频评分（TES & PCS）训练与评估脚本，基于 Video Swin Transformer + MLP（支持混合精度训练，节省显存）

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2
from mmaction.apis import init_recognizer
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import mean_squared_error  # 新增：导入MSE计算函数

# ---------- 硬编码模型配置和权重路径 ----------
root = '/root/autodl-tmp/Video-Swin-Transformer-master'  # 模型仓库根目录
config_file = os.path.join(
    root,
    'configs/recognition/swin/swin_base_patch244_window877_kinetics600_22k.py'
)
checkpoint_file = os.path.join(
    root,
    'swin_base_patch244_window877_kinetics600_22k.pth'
)
# -----------------------------------------------

# ---------- 硬编码数据集和视频路径 ----------
dataset_root = '/root/autodl-tmp/dataset'  # 存放 train_dataset.txt, test_dataset.txt
video_dir = '/root/autodl-tmp/videos'     # 存放所有视频文件 (.mp4, .mov)
# -----------------------------------------------


def setup_seed(seed):
    """
    设置全局随机种子，保证实验可复现
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class VideoDataset(Dataset):
    """
    自定义视频数据集（支持 TES/PCS Min-Max归一化）
    """
    def __init__(self, list_file, transform=None, clip_len=32, 
                 tes_min=0.0, tes_max=1.0,  # 新增：训练集 TES 最小/最大值
                 pcs_min=0.0, pcs_max=1.0): # 新增：训练集 PCS 最小/最大值
        self.clip_len = clip_len
        self.transform = transform
        self.samples = []
        # 保存Min-Max归一化参数
        self.tes_min = tes_min
        self.tes_max = tes_max
        self.pcs_min = pcs_min
        self.pcs_max = pcs_max
        with open(list_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                video_id, tes_score, pcs_score = parts[0], float(parts[1]), float(parts[2])
                mp4 = os.path.join(video_dir, f"{video_id}.mp4")
                mov = os.path.join(video_dir, f"{video_id}.mov")
                if os.path.isfile(mp4):
                    path = mp4
                elif os.path.isfile(mov):
                    path = mov
                else:
                    print(f"Warning: 视频 {video_id}.mp4/.mov 未找到，跳过")
                    continue
                self.samples.append((path, tes_score, pcs_score))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, tes, pcs = self.samples[idx]
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            raise RuntimeError(f"无法读取视频 {path}")
        indices = np.linspace(0, total - 1, self.clip_len, dtype=int)
        frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            if self.transform:
                img = self.transform(img)
            frames.append(img)
        cap.release()
        vid = torch.stack(frames, dim=1)
        # Min-Max归一化（防止除零，分母加1e-8）
        normalized_tes = (tes - self.tes_min) / (self.tes_max - self.tes_min + 1e-8)
        normalized_pcs = (pcs - self.pcs_min) / (self.pcs_max - self.pcs_min + 1e-8)
        target = torch.tensor([normalized_tes, normalized_pcs], dtype=torch.float32)
        return vid, target


class E2EModel(nn.Module):
    """
    端到端模型：Video Swin Transformer + 双独立预测头（TES/PCS 分别建模）
    """
    def __init__(self, device, clip_len, mlp_hidden=1024, dropout=0.1):  # 默认 dropout 率调整为 0.1
        super().__init__()
        self.device = device
        recognizer = init_recognizer(config_file, checkpoint_file, device=device)
        self.backbone = recognizer.backbone
        # 冻结预训练骨干网络参数
        for param in self.backbone.parameters():
            param.requires_grad = False  # 新增：冻结骨干网络参数
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # 动态获取 backbone 输出维度（未修改）
        with torch.no_grad():
            dummy = torch.zeros(1, 3, clip_len, 224, 224).to(device)
            out = self.backbone(dummy)
            feat = out[0] if isinstance(out, (list, tuple)) else out
            dim = feat.shape[1]
        
        # 独立 TES 预测头（增加到 4 层 Linear）
        self.tes_head = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            # 新增第 2 个隐藏层
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            # 原第 2 个隐藏层调整为第 3 层
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.LayerNorm(mlp_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            # 新增第 4 个隐藏层（输出前的过渡层）
            nn.Linear(mlp_hidden // 2, mlp_hidden // 4),
            nn.LayerNorm(mlp_hidden // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden // 4, 1)  # TES 输出 1 维
        )

        # 独立 PCS 预测头（与 TES 头结构对称，同步增加层数）
        self.pcs_head = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            # 新增第 2 个隐藏层
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            # 原第 2 个隐藏层调整为第 3 层
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.LayerNorm(mlp_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            # 新增第 4 个隐藏层（输出前的过渡层）
            nn.Linear(mlp_hidden // 2, mlp_hidden // 4),
            nn.LayerNorm(mlp_hidden // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden // 4, 1)  # PCS 输出 1 维
        )

    def forward(self, x):
        x = x.to(self.device)
        out = self.backbone(x)
        feat = out[0] if isinstance(out, (list, tuple)) else out
        feat = self.pool(feat).view(feat.size(0), -1)  # 展平为 [B, dim]
        tes_pred = self.tes_head(feat)  # [B, 1]
        pcs_pred = self.pcs_head(feat)  # [B, 1]
        return torch.cat([tes_pred, pcs_pred], dim=1)  # 合并为 [B, 2] 保持输出形状与原一致


def train_epoch(model, loader, criterion, optimizer, scaler, device):
    """
    单个 epoch 训练，使用混合精度
    """
    model.train()
    total_loss = 0
    for vids, tgts in tqdm(loader, desc='Train', leave=False):
        vids, tgts = vids.to(device), tgts.to(device)
        optimizer.zero_grad()
        with autocast():
            preds = model(vids)
            loss = criterion(preds, tgts)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * vids.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device, tes_min, tes_max, pcs_min, pcs_max):  # 新增Min-Max参数
    model.eval()
    all_preds, all_tgts = [], []
    with torch.no_grad():
        for vids, tgts in tqdm(loader, desc='Eval', leave=False):
            vids = vids.to(device)
            with autocast():
                preds = model(vids).cpu().numpy()  # 模型输出是归一化值
            all_preds.append(preds)
            all_tgts.append(tgts.numpy())  # 真实标签是归一化值

    # 合并所有预测和真实值
    p = np.concatenate(all_preds)
    t = np.concatenate(all_tgts)

    # 反Min-Max归一化（恢复原始尺度）
    p_tes = p[:, 0] * (tes_max - tes_min) + tes_min  # 预测 TES 原始值
    p_pcs = p[:, 1] * (pcs_max - pcs_min) + pcs_min  # 预测 PCS 原始值
    t_tes = t[:, 0] * (tes_max - tes_min) + tes_min  # 真实 TES 原始值
    t_pcs = t[:, 1] * (pcs_max - pcs_min) + pcs_min  # 真实 PCS 原始值

    # 计算SRCC和MSE
    tes_srcc = spearmanr(t_tes, p_tes).correlation
    pcs_srcc = spearmanr(t_pcs, p_pcs).correlation
    tes_mse = mean_squared_error(t_tes, p_tes)
    pcs_mse = mean_squared_error(t_pcs, p_pcs)
    
    return tes_srcc, pcs_srcc, tes_mse, pcs_mse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip-len', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=2, help='缩小 batch_size 以节省显存')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--mlp-hidden', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    setup_seed(42)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_list = os.path.join(dataset_root, 'train_dataset.txt')
    test_list  = os.path.join(dataset_root, 'test_dataset.txt')

    # 初始加载训练集以计算Min-Max统计量
    temp_train_ds = VideoDataset(train_list, transform=transform, clip_len=args.clip_len)
    train_tes = [sample[1] for sample in temp_train_ds.samples]  # 原始 TES 分数
    train_pcs = [sample[2] for sample in temp_train_ds.samples]  # 原始 PCS 分数
    tes_min, tes_max = np.min(train_tes), np.max(train_tes)
    pcs_min, pcs_max = np.min(train_pcs), np.max(train_pcs)

    # 重新初始化数据集（传入Min-Max参数）
    train_ds = VideoDataset(
        train_list, 
        transform=transform, 
        clip_len=args.clip_len,
        tes_min=tes_min, 
        tes_max=tes_max,
        pcs_min=pcs_min, 
        pcs_max=pcs_max
    )
    test_ds = VideoDataset(
        test_list, 
        transform=transform, 
        clip_len=args.clip_len,
        tes_min=tes_min,  # 测试集使用训练集的Min-Max
        tes_max=tes_max,
        pcs_min=pcs_min,
        pcs_max=pcs_max
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 新增：打印训练集前5个样本信息（路径、TES、PCS）
    print("\n===== 训练集前5个样本 =====")
    for i in range(min(5, len(train_ds.samples))):
        path, tes, pcs = train_ds.samples[i]
        print(f"样本 {i+1}: 视频路径={path}, TES分数={tes:.2f}, PCS分数={pcs:.2f}")

    # 新增：打印测试集前5个样本信息（路径、TES、PCS）
    print("\n===== 测试集前5个样本 =====")
    for i in range(min(5, len(test_ds.samples))):
        path, tes, pcs = test_ds.samples[i]
        print(f"样本 {i+1}: 视频路径={path}, TES分数={tes:.2f}, PCS分数={pcs:.2f}")

    model = E2EModel(device, args.clip_len, args.mlp_hidden, args.dropout).to(device)
    criterion = nn.MSELoss()
    # 修改优化器参数：仅优化MLP头参数
    optimizer = optim.AdamW(
        list(model.tes_head.parameters()) + list(model.pcs_head.parameters()),  # 仅包含MLP头参数
        lr=args.lr
    )
    scaler = GradScaler()

    losses = []
    for epoch in range(1, args.epochs+1):
        loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        losses.append(loss)
        print(f"Epoch {epoch}/{args.epochs}  Loss: {loss:.4f}")
        
        # 每个epoch执行评估（传递Min-Max参数）
        tes_srcc, pcs_srcc, tes_mse, pcs_mse = evaluate(
            model, test_loader, device,
            tes_min=tes_min, tes_max=tes_max,
            pcs_min=pcs_min, pcs_max=pcs_max
        )
        print(f"Epoch {epoch} TEST - TES SRCC: {tes_srcc:.4f}, MSE: {tes_mse:.4f}; PCS SRCC: {pcs_srcc:.4f}, MSE: {pcs_mse:.4f}")

    # 最终评估
    tes_srcc, pcs_srcc, tes_mse, pcs_mse = evaluate(
        model, test_loader, device,
        tes_min=tes_min, tes_max=tes_max,
        pcs_min=pcs_min, pcs_max=pcs_max
    )
    print(f"FINAL TEST - TES SRCC: {tes_srcc:.4f}, MSE: {tes_mse:.4f}; PCS SRCC: {pcs_srcc:.4f}, MSE: {pcs_mse:.4f}")


if __name__ == '__main__':
    main()
