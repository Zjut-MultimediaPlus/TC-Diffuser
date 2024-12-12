
import torch
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 加载保存的二进制文件1

his_pos = torch.load('old6_his_pos.pt') #torch.Size([6,3, 8, 2]) 历史的8个点
eval_predicted_trajs = torch.load('old6_predicted_trajs.pt') #torch.Size([6, 3, 6, 4, 2])
eval_gt_trajs = torch.load('old6_gt_trajs.pt') #torch.Size([6, 3, 4, 2])  [迭代次数，B，T4，2]

# 选一个B
his_pos = his_pos[:,1,:,:] #[6, 8, 2]
eval_predicted_trajs = eval_predicted_trajs[:,1,:,:,:] #[6, 6, 4, 2]
eval_gt_trajs = eval_gt_trajs[:,1,:,:] #[6, 4, 2]


for die in [20,40,80,100]: #迭代次数20,40,80,100

    pred = eval_predicted_trajs[die] #[6, 4, 2]
    gt = eval_gt_trajs[die] #[4,2]

    mse = 0
    for i in range(pred.size(0)): #6条轨迹
        each_traj = pred[i]  # 获取每个[4, 2]的子张量
        difference = each_traj - gt
        squared_difference = difference.pow(2)
        mse = mse + squared_difference.mean()
    mse = mse/pred.size(0)
    mse = mse/10
    print("迭代次数：",die ," 不确定性得分：",mse)




