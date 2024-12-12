import torch
import torch.nn.utils.rnn as rnn
from enum import Enum
import functools
import numpy as np
import math


class ModeKeys(Enum):
    TRAIN = 1
    EVAL = 2
    PREDICT = 3


def cyclical_lr(stepsize, min_lr=3e-4, max_lr=3e-3, decay=1.):
    # Lambda function to calculate the LR
    lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize) * decay**it

    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x))

    return lr_lambda


def to_one_hot(labels, n_labels):
    return torch.eye(n_labels, device=labels.device)[labels]


def exp_anneal(anneal_kws):
    device = anneal_kws['device']
    start = torch.tensor(anneal_kws['start'], device=device)
    finish = torch.tensor(anneal_kws['finish'], device=device)
    rate = torch.tensor(anneal_kws['rate'], device=device)
    return lambda step: finish - (finish - start)*torch.pow(rate, torch.tensor(step, dtype=torch.float, device=device))


def sigmoid_anneal(anneal_kws):
    device = anneal_kws['device']
    start = torch.tensor(anneal_kws['start'], device=device)
    finish = torch.tensor(anneal_kws['finish'], device=device)
    center_step = torch.tensor(anneal_kws['center_step'], device=device, dtype=torch.float)
    steps_lo_to_hi = torch.tensor(anneal_kws['steps_lo_to_hi'], device=device, dtype=torch.float)
    return lambda step: start + (finish - start)*torch.sigmoid((torch.tensor(float(step), device=device) - center_step) * (1./steps_lo_to_hi))


class CustomLR(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        super(CustomLR, self).__init__(optimizer, lr_lambda, last_epoch)

    def get_lr(self):
        return [lmbda(self.last_epoch)
                for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]


def mutual_inf_mc(x_dist):
    dist = x_dist.__class__
    H_y = dist(probs=x_dist.probs.mean(dim=0)).entropy()
    return (H_y - x_dist.entropy().mean(dim=0)).sum()

# 这段代码是一个在变长序列上运行LSTM模型的函数。下面是函数的主要步骤：
#
# 1. 接收输入参数，包括LSTM模型（lstm_module）、原始序列（original_seqs）、下界索引（lower_indices）、上界索引（upper_indices）和总长度（total_length）。
# 2. 检查是否提供了下界索引、上界索引和总长度。如果没有提供，则设置默认值。
# 3. 根据下界索引和上界索引，截取原始序列中的子序列，并形成一个列表。
# 4. 使用`torch.nn.utils.rnn.pack_sequence`函数将子序列打包成一个压缩的序列对象（packed sequence），并保留原始序列的顺序。
# 这个步骤主要是为了处理变长序列。
# 5. 将打包的序列对象输入到LSTM模型中，得到输出序列和最后一个时间步的隐藏状态和细胞状态。
# 6. 使用`torch.nn.utils.rnn.pad_packed_sequence`函数将压缩的序列对象解压缩，并将其填充为与总长度相匹配的形状。
# 7. 返回输出序列和最后一个时间步的隐藏状态和细胞状态。
#
# 该函数的作用是在变长序列上应用LSTM模型，并处理序列的变长情况，以便在处理序列数据时进行有效的批处理操作。

#有可能这边用的是同一份模型的！
def run_lstm_on_variable_length_seqs(lstm_module, original_seqs, lower_indices=None, upper_indices=None, total_length=None):
    bs, tf = original_seqs.shape[:2]  #original_seqs:torch.Size([256, 8, 12]) 提取了original_seqs张量的前两个维度 bs:255  tf:8
    if lower_indices is None:
        lower_indices = torch.zeros(bs, dtype=torch.int)
    if upper_indices is None:
        upper_indices = torch.ones(bs, dtype=torch.int) * (tf - 1) #全7
    if total_length is None:
        total_length = max(upper_indices) + 1 #8
    # This is done so that we can just pass in self.prediction_timesteps
    # (which we want to INCLUDE, so this will exclude the next timestep).
    inclusive_break_indices = upper_indices + 1 #torch.Size([255]) 全8

    pad_list = list()
    for i, seq_len in enumerate(inclusive_break_indices):
        pad_list.append(original_seqs[i, lower_indices[i]:seq_len])  #一个列表 256个torch.Size([8, 12]) 与original_seqs一致

    packed_seqs = rnn.pack_sequence(pad_list, enforce_sorted=False) # pad_list：长255 torch.Size([8, 6])
    packed_output, (h_n, c_n) = lstm_module(packed_seqs)  #packed_seqs和packed_output的0的维度 (17xx,6)--(17xx,128)
    output, _ = rnn.pad_packed_sequence(packed_output,
                                        batch_first=True,
                                        total_length=total_length)

    return output, (h_n, c_n) #torch.Size([255, 8, 128]) torch.Size([1, 255, 128]) torch.Size([1, 255, 128])

def run_lstm_on_variable_length_seqs2(lstm_module, original_seqs, encoded_gph_data, lower_indices=None, upper_indices=None, total_length=None):
    original_seqs = original_seqs + encoded_gph_data*0.01
    bs, tf = original_seqs.shape[:2]  #original_seqs:torch.Size([256, 8, 12]) 提取了original_seqs张量的前两个维度 bs:255  tf:8
    if lower_indices is None:
        lower_indices = torch.zeros(bs, dtype=torch.int)
    if upper_indices is None:
        upper_indices = torch.ones(bs, dtype=torch.int) * (tf - 1) #全7
    if total_length is None:
        total_length = max(upper_indices) + 1 #8
    # This is done so that we can just pass in self.prediction_timesteps
    # (which we want to INCLUDE, so this will exclude the next timestep).
    inclusive_break_indices = upper_indices + 1 #torch.Size([255]) 全8

    pad_list = list()
    for i, seq_len in enumerate(inclusive_break_indices):
        pad_list.append(original_seqs[i, lower_indices[i]:seq_len])  #一个列表 256个torch.Size([8, 12]) 与original_seqs一致

    packed_seqs = rnn.pack_sequence(pad_list, enforce_sorted=False) # pad_list：长255 torch.Size([8, 6])
    packed_output, (h_n, c_n) = lstm_module(packed_seqs)  #packed_seqs和packed_output的0的维度 (17xx,6)--(17xx,128)
    output, _ = rnn.pad_packed_sequence(packed_output,
                                        batch_first=True,
                                        total_length=total_length)

    return output, (h_n, c_n) #torch.Size([255, 8, 128]) torch.Size([1, 255, 128]) torch.Size([1, 255, 128])


def extract_subtensor_per_batch_element(tensor, indices):
    batch_idxs = torch.arange(start=0, end=len(indices))

    batch_idxs = batch_idxs[~torch.isnan(indices)]
    indices = indices[~torch.isnan(indices)]
    if indices.size == 0:
        return None
    else:
        indices = indices.long()
    if tensor.is_cuda:
        batch_idxs = batch_idxs.to(tensor.get_device())
        indices = indices.to(tensor.get_device())
    return tensor[batch_idxs, indices]


def unpack_RNN_state(state_tuple):
    # PyTorch returned LSTM states have 3 dims:
    # (num_layers * num_directions, batch, hidden_size)

    state = torch.cat(state_tuple, dim=0).permute(1, 0, 2)
    # Now state is (batch, 2 * num_layers * num_directions, hidden_size)

    state_size = state.size()
    return torch.reshape(state, (-1, state_size[1] * state_size[2]))


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


# using wonder's beautiful simplification:
# https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))
