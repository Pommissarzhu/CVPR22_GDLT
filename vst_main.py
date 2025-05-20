from scipy.special import chebyc
import torch
import numpy as np
import options
from datasets import RGDataset
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models import model, loss
import os
from torch import nn
from torchvision import transforms

import train
from test import test_epoch

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

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optim(model, args):
    if args.optim == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'rmsprop':
        optim = torch.optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise Exception("Unknown optimizer")
    return optim


def get_scheduler(optim, args):
    if args.lr_decay is not None:
        if args.lr_decay == 'cos':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, T_max=args.epoch - args.warmup, eta_min=args.lr * args.decay_rate)
        elif args.lr_decay == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[args.epoch - 30], gamma=args.decay_rate)
        else:
            raise Exception("Unknown Scheduler")
    else:
        scheduler = None
    return scheduler

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

if __name__ == '__main__':
    args = options.parser.parse_args()
    setup_seed(0)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dataset_root = "dataetroot"

    '''
    1. load data
    '''
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
    print('=============Load dataset successfully=============')

    '''
    2. load model
    '''
    model = model.VST_GDLT(args.in_dim, args.hidden_dim, args.n_head, args.n_encoder,
                       args.n_decoder, args.n_query, args.dropout, 
                       device=device, config_file=config_file, checkpoint_file=checkpoint_file).to(device)
    loss_fn = loss.LossFun(args.alpha, args.margin)
    train_fn = train.train_epoch
    if args.ckpt is not None:
        checkpoint = torch.load('./ckpt/' + args.ckpt + '.pkl')
        model.load_state_dict(checkpoint)
    print('=============Load model successfully=============')

    print(args)

    '''
    test mode
    '''
    if args.test:
        test_loss, coef = test_epoch(0, model, test_loader, None, device, args)
        print('Test Loss: {:.4f}\tTest Coef: {:.3f}'.format(test_loss, coef))
        raise SystemExit

    '''
    3. record
    '''
    if not os.path.exists("./ckpt/"):
        os.makedirs("./ckpt/")
    if not os.path.exists("./logs/" + args.model_name):
        os.makedirs("./logs/" + args.model_name)
    logger = SummaryWriter(os.path.join('./logs/', args.model_name))
    best_coef, best_epoch = -1, -1
    final_train_loss, final_train_coef, final_test_loss, final_test_coef = 0, 0, 0, 0

    '''
    4. train
    '''
    optim = get_optim(model, args)
    scheduler = get_scheduler(optim, args)
    print('=============Begin training=============')
    if args.warmup:
        warmup = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda t: t / args.warmup)
    else:
        warmup = None

    for epc in range(args.epoch):
        if args.warmup and epc < args.warmup:
            warmup.step()
        # print(optim.state_dict()['param_groups'][0]['lr'])
        avg_loss, train_coef = train_fn(epc, model, loss_fn, train_loader, optim, logger, device, args)
        if scheduler is not None and (args.lr_decay != 'cos' or epc >= args.warmup):
            scheduler.step()
        test_loss, test_coef = test_epoch(epc, model, test_loader, logger, device, args)
        if test_coef > best_coef:
            best_coef, best_epoch = test_coef, epc
            torch.save(model.state_dict(), './ckpt/' + args.model_name + '_best.pkl')

        print('Epoch: {}\tLoss: {:.4f}\tTrain Coef: {:.3f}\tTest Loss: {:.4f}\tTest Coef: {:.3f}'
              .format(epc, avg_loss, train_coef, test_loss, test_coef))
        if epc == args.epoch - 1:
            final_train_loss, final_train_coef, final_test_loss, final_test_coef = \
                avg_loss, train_coef, test_loss, test_coef
    torch.save(model.state_dict(), './ckpt/' + args.model_name + '.pkl')
    print('Best Test Coef: {:.3f}\tBest Test Eopch: {}'.format(best_coef, best_epoch))
