import torch
from torch.nn import Module, Linear
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import torch.nn as nn
import math

def reparameterize_gaussian(mean, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn(std.size()).to(mean)
    return mean + std * eps


def gaussian_entropy(logvar):
    const = 0.5 * float(logvar.size(1)) * (1. + np.log(np.pi * 2))
    ent = 0.5 * logvar.sum(dim=1, keepdim=False) + const
    return ent


def standard_normal_logprob(z):
    dim = z.size(-1)
    log_z = -0.5 * dim * np.log(2 * np.pi)
    return log_z - z.pow(2) / 2


def truncated_normal_(tensor, mean=0, std=1, trunc_std=2):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < trunc_std) & (tmp > -trunc_std)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

# pos_emb 是位置编码器，用于【为输入数据增加位置信息】，以帮助 Transformer 模型捕捉输入数据中的位置关系。
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        #d_model 是【输入数据的维度】，也是位置编码的维度。
        # dropout 是在位置编码后是否应用 dropout 操作的概率，默认为 0.1。
        # max_len 是输入序列的最大长度，在位置编码中使用，用于控制位置信息的范围。
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        #将位置编码 pe 加到输入数据 x 上，实现位置信息的增强。
        x = x + self.pe[: x.size(0), :]
        #最后，对增强后的数据应用 dropout 操作，以防止过拟合。
        return self.dropout(x)
    #PositionalEncoding 类用于为输入数据增加位置编码，
    # 通过 sine 和 cosine 函数生成位置编码，然后将位置编码加到输入数据上，
    # 以增强模型对输入序列位置信息的感知能力。
    # 同时，可以应用 dropout 操作来提高模型的泛化能力。
    # 这在 Transformer 等模型中特别有用，
    # 因为这些模型没有显式地处理序列的顺序信息，而是通过位置编码来引入序列的位置信息。


#其中，dim_in 是输入的维度，dim_out 是输出的维度，dim_ctx 是上下文的维度。
class ConcatSquashLinear(Module):
    #                  (2,2*context_dim,context_dim+3)
    def __init__(self, dim_in, dim_out, dim_ctx): #4, 256, 84
        #point_dim,2*context_dim,context_dim+3
        super(ConcatSquashLinear, self).__init__()
        #三个类成员变量 _layer，_hyper_bias和_hyper_gate，它们分别表示线性层、门控偏置和门控权重
        self._layer = Linear(dim_in, dim_out)#4--512   256--128
        self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = Linear(dim_ctx, dim_out)
#     self.concat1(ctx_emb,x)
    def forward(self, ctx, x): #(B, 1, 84) (B,4,4)
        gate = torch.sigmoid(self._hyper_gate(ctx)) #(B,1,256)
        bias = self._hyper_bias(ctx) #(B,1,256)
        a = self._layer(x) #(B,4,256)
        ret = a * gate + bias #x:torch.Size([256, 4, 4])  
        return ret  #torch.Size([256, 4, 256])

class ConcatTransformerLinear(Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatTransformerLinear, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim_in, nhead=8)
        #self._layer = Linear(dim_in, dim_out)
        self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        # x: (B*12*2)
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret = self.encoder_layer(x) * gate + bias
        return ret


def get_linear_scheduler(optimizer, start_epoch, end_epoch, start_lr, end_lr):
    def lr_func(epoch):
        if epoch <= start_epoch:
            return 1.0
        elif epoch <= end_epoch:
            total = end_epoch - start_epoch
            delta = epoch - start_epoch
            frac = delta / total
            return (1-frac) * 1.0 + frac * (end_lr / start_lr)
        else:
            return end_lr / start_lr
    return LambdaLR(optimizer, lr_lambda=lr_func)

def lr_func(epoch):
    if epoch <= start_epoch:
        return 1.0
    elif epoch <= end_epoch:
        total = end_epoch - start_epoch
        delta = epoch - start_epoch
        frac = delta / total
        return (1-frac) * 1.0 + frac * (end_lr / start_lr)
    else:
        return end_lr / start_lr
