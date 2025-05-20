from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
import copy
from models.transformer import Transformer
from mmaction.apis import init_recognizer


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        try:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0)
        except AttributeError:
            pass


# 主力model
class GDLT(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_head, n_encoder, n_decoder, n_query, dropout):
        super(GDLT, self).__init__()
        self.in_proj = nn.Sequential(
            nn.Conv1d(kernel_size=1, in_channels=in_dim, out_channels=in_dim // 2),
            nn.BatchNorm1d(in_dim // 2),
            nn.ReLU(),
            nn.Conv1d(kernel_size=1, in_channels=in_dim // 2, out_channels=hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )

        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=n_head,
            num_encoder_layers=n_encoder,
            num_decoder_layers=n_decoder,
            dim_feedforward=3 * hidden_dim,
            batch_first=True,
            dropout=dropout
        )

        self.prototype = nn.Embedding(n_query, hidden_dim)

        self.weight = torch.linspace(0, 1, n_query, requires_grad=False).cuda()
        print(self.weight)
        self.regressor = nn.Linear(hidden_dim, n_query)

    def forward(self, x):
        # x (b, t, c)
        b, t, c = x.shape
        x = self.in_proj(x.transpose(1, 2)).transpose(1, 2)

        q = self.prototype.weight.unsqueeze(0).repeat(b, 1, 1)
        encode_x = self.transformer.encoder(x)
        q1 = self.transformer.decoder(q, encode_x)

        s = self.regressor(q1)  # (b, n, n)
        s = torch.diagonal(s, dim1=-2, dim2=-1)  # (b, n)
        norm_s = torch.sigmoid(s)
        norm_s = norm_s / torch.sum(norm_s, dim=1, keepdim=True)
        out = torch.sum(self.weight.unsqueeze(0).repeat(b, 1) * norm_s, dim=1)

        return {'output': out, 'embed': q1}

class VST_GDLT(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_head, n_encoder, n_decoder, n_query, dropout, device, config_file, checkpoint_file):
        super(VST_GDLT, self).__init__()

        self.device = device
        recognizer = init_recognizer(config_file, checkpoint_file, device=device)
        self.vst_backbone = recognizer.backbone
        # 冻结预训练骨干网络参数
        for param in self.vst_backbone.parameters():
            param.requires_grad = False  # 新增：冻结骨干网络参数

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.in_proj = nn.Sequential(
            nn.Conv1d(kernel_size=1, in_channels=in_dim, out_channels=in_dim // 2),
            nn.BatchNorm1d(in_dim // 2),
            nn.ReLU(),
            nn.Conv1d(kernel_size=1, in_channels=in_dim // 2, out_channels=hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )

        self.transformer = Transformer(
            d_model=hidden_dim,
            nhead=n_head,
            num_encoder_layers=n_encoder,
            num_decoder_layers=n_decoder,
            dim_feedforward=3 * hidden_dim,
            batch_first=True,
            dropout=dropout
        )

        self.prototype = nn.Embedding(n_query, hidden_dim)

        self.weight = torch.linspace(0, 1, n_query, requires_grad=False).cuda()
        print(self.weight)
        self.regressor = nn.Linear(hidden_dim, n_query)

    def forward(self, x):
        x = x.to(self.device)  # 将输入数据移动到指定设备上
        with torch.no_grad():  # 新增：在测试模式下不计算梯度
            x = self.vst_backbone(x)  # 使用预训练骨干网络提取特征
            x = self.pool(x).squeeze(3).squeeze(3)  # 调整维度以匹配输入要求
        
        # x (b, t, c)
        b, t, c = x.shape
        x = self.in_proj(x.transpose(1, 2)).transpose(1, 2)

        q = self.prototype.weight.unsqueeze(0).repeat(b, 1, 1)
        encode_x = self.transformer.encoder(x)
        q1 = self.transformer.decoder(q, encode_x)

        s = self.regressor(q1)  # (b, n, n)
        s = torch.diagonal(s, dim1=-2, dim2=-1)  # (b, n)
        norm_s = torch.sigmoid(s)
        norm_s = norm_s / torch.sum(norm_s, dim=1, keepdim=True)
        out = torch.sum(self.weight.unsqueeze(0).repeat(b, 1) * norm_s, dim=1)

        return {'output': out, 'embed': q1}