import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import binary_dilation
from scipy.stats import gaussian_kde
from .trajectory_utils import prediction_output_to_trajectories
import torch
#import visualization
from matplotlib import pyplot as plt
import pdb

#  <class 'tuple'>: (1, 20, 4, 2)----<class 'tuple'>: (4, 2)
def compute_ade(predicted_trajs, gt_traj):
    #pdb.set_trace() error:<class 'tuple'>: (1, 20, 4)
    error = np.linalg.norm(predicted_trajs - gt_traj, axis=-1) #np.linalg.norm函数在最后一个轴上计算欧氏范数来实现
    ade = np.mean(error, axis=-1) #函数通过沿最后一个轴（axis=-1）取平均值来计算每个轨迹的平均误差。 <class 'tuple'>: (1, 20)
    # 这将得到一个形状为[batch_size]的1D numpy数组，其中每个元素表示单个轨迹的ADE。
    return ade.flatten() #使用flatten()将ADE值展平为一个1D数组，然后将其作为函数的返回值。 <class 'tuple'>: (20,)

#  <class 'tuple'>: (1, 20, 4, 2)----<class 'tuple'>: (4, 2)
def compute_ade_x_y_traj(predicted_trajs, gt_traj):
    predicted_trajs = predicted_trajs[:, :, :, 0:2]
    gt_traj = gt_traj[:, 0:2]
    num_of_trajs = predicted_trajs.shape[1] #20条轨迹/6条轨迹
    # 将tuple1复制为一个列表，重复20次
    tuples = [gt_traj] * num_of_trajs
    gt_traj_copy = np.array(tuples) #<class 'tuple'>: (20, 4, 2)

    original_list = np.array(predicted_trajs)  # 将列表转换为NumPy数组
    predicted_trajs = original_list.reshape(num_of_trajs, predicted_trajs.shape[2], 2)  # 重新调整数组的形状 <class 'tuple'>: (20, 4, 2)

    # 0经度  1纬度  实际经纬度 预测值
    predicted_trajs[:, :, 0] = predicted_trajs[:, :, 0] / 10 * 500 + 1800
    predicted_trajs[:, :, 1] = predicted_trajs[:, :, 1] / 6 * 300
    #实际经纬度 ground truth
    gt_traj_copy[:, :, 0] = gt_traj_copy[:, :, 0] / 10 * 500 + 1800
    gt_traj_copy[:, :, 1] = gt_traj_copy[:, :, 1] / 6 * 300

    interpolation = predicted_trajs - gt_traj_copy #(20, 4, 2) 差值
    # 实际的距离上的差值
    interpolation[:, :, 0] = (interpolation[:, :, 0] / 10) * 111
    interpolation[:, :, 1] = (interpolation[:, :, 1] / 10) * 111 * np.cos(gt_traj_copy[:, :, 1] / 10 * np.pi / 180)
    #
    interpolation = torch.from_numpy(interpolation) #torch.Size([6, 4, 2])

    interpolation = interpolation ** 2

    loss = torch.sqrt(interpolation[:,:,0]+interpolation[:,:,1]) #torch.Size([6, 4])
    sum_tensor = torch.sum(loss, dim=-1) #(20) 4个误差求和 torch.Size([20])

    # 使用torch.min()函数找到最小值及其下标
    min_value, min_index = torch.min(sum_tensor, dim = 0)

    return min_value, min_index, loss[min_index.item(),:], predicted_trajs, gt_traj_copy #-,-,实际误差最小的一条轨迹的误差值（4）

# <class 'tuple'>: (1, 6, 4, 4)
def compute_ade_x_y_intensity(predicted_trajs, gt_traj):
    predicted_trajs = predicted_trajs[:,:,:,2:4]
    gt_traj = gt_traj[:, 2:4]

    num_of_trajs = predicted_trajs.shape[1] #20条轨迹/6条轨迹
    # 将tuple1复制为一个列表，重复20次
    tuples = [gt_traj] * num_of_trajs
    gt_traj_copy = np.array(tuples) #<class 'tuple'>: (20, 4, 2)

    original_list = np.array(predicted_trajs)  # 将列表转换为NumPy数组
    predicted_trajs = original_list.reshape(num_of_trajs, predicted_trajs.shape[2], 2)  # 重新调整数组的形状 <class 'tuple'>: (20, 4, 2)

    # 0强度 1风速
    predicted_trajs[:, :, 0] = predicted_trajs[:, :, 0] * 50 + 960
    predicted_trajs[:, :, 1] = predicted_trajs[:, :, 1] * 25 + 40
    # 实际强度+风速 ground truth
    gt_traj_copy[:, :, 0] = gt_traj_copy[:, :, 0] * 50 + 960
    gt_traj_copy[:, :, 1] = gt_traj_copy[:, :, 1] * 25 + 40


    interpolation = predicted_trajs - gt_traj_copy #(6, 4, 2) 差值 (,T,)

    interpolation = torch.from_numpy(interpolation)


    loss_intensity = interpolation[:, :, 0] #torch.
    loss_wind = interpolation[:, :, 1]

    abs_inten = torch.abs(loss_intensity)
    abs_wind = torch.abs(loss_wind)


    sum_tensor_intensity = torch.sum(abs_inten, dim=-1) #(20) 4个时刻的误差求和 torch.Size([20])
    sum_tensor_wind = torch.sum(abs_wind, dim=-1)  # (20) 4个误差求和 torch.Size([20])

    # 使用torch.min()函数找到最小值及其下标
    min_value_intensity, min_index_intensity = torch.min(sum_tensor_intensity, dim = 0)
    min_value_wind, min_index_wind = torch.min(sum_tensor_wind, dim=0)


    return min_value_intensity, min_value_wind, min_index_intensity, min_index_wind, \
           loss_intensity[min_index_intensity.item(),:], loss_wind[min_index_wind.item(),:],\
           predicted_trajs, gt_traj_copy #-,-,实际误差最小的一条轨迹的误差值（4）

def compute_ade_x_y_traj_each_time(predicted_trajs, gt_traj):
    predicted_trajs = predicted_trajs[:, :, :, 0:2]
    gt_traj = gt_traj[:, 0:2]
    num_of_trajs = predicted_trajs.shape[1] #20条轨迹/6条轨迹
    # 将tuple1复制为一个列表，重复20次
    tuples = [gt_traj] * num_of_trajs
    gt_traj_copy = np.array(tuples) #<class 'tuple'>: (20, 4, 2)

    original_list = np.array(predicted_trajs)  # 将列表转换为NumPy数组
    predicted_trajs = original_list.reshape(num_of_trajs, predicted_trajs.shape[2], 2)  # 重新调整数组的形状 <class 'tuple'>: (20, 4, 2)

    # 0经度  1纬度  实际经纬度 预测值
    predicted_trajs[:, :, 0] = predicted_trajs[:, :, 0] / 10 * 500 + 1800
    predicted_trajs[:, :, 1] = predicted_trajs[:, :, 1] / 6 * 300
    #实际经纬度 ground truth
    gt_traj_copy[:, :, 0] = gt_traj_copy[:, :, 0] / 10 * 500 + 1800
    gt_traj_copy[:, :, 1] = gt_traj_copy[:, :, 1] / 6 * 300

    interpolation = predicted_trajs - gt_traj_copy #(20, 4, 2) 差值
    # 实际的距离上的差值
    interpolation[:, :, 0] = (interpolation[:, :, 0] / 10) * 111
    interpolation[:, :, 1] = (interpolation[:, :, 1] / 10) * 111 * np.cos(gt_traj_copy[:, :, 1] / 10 * np.pi / 180)
    #
    interpolation = torch.from_numpy(interpolation) #torch.Size([20条数, 4, 2经纬度])

    interpolation = interpolation ** 2

    loss = torch.sqrt(interpolation[:,:,0]+interpolation[:,:,1]) #torch.Size([20条, 4个时刻])
    min_value_of_4_time = torch.tensor([0.0, 0.0, 0.0, 0.0])
    min_index = torch.tensor([0, 0, 0, 0])
    for t in range(loss.size(1)):
        min_value_of_4_time[t], min_index[t] = torch.min(loss[:, t], dim=0)

    sum_min_value = torch.sum(min_value_of_4_time, dim=-1)

    return sum_min_value, min_index, min_value_of_4_time, predicted_trajs, gt_traj_copy #-,-,实际误差最小的一条轨迹的误差值（4）

# <class 'tuple'>: (1, 6, 4, 4)
def compute_ade_x_y_intensity_each_time(predicted_trajs, gt_traj):
    predicted_trajs = predicted_trajs[:,:,:,2:4]
    gt_traj = gt_traj[:, 2:4]

    num_of_trajs = predicted_trajs.shape[1] #20条轨迹/6条轨迹
    # 将tuple1复制为一个列表，重复20次
    tuples = [gt_traj] * num_of_trajs
    gt_traj_copy = np.array(tuples) #<class 'tuple'>: (20, 4, 2)

    original_list = np.array(predicted_trajs)  # 将列表转换为NumPy数组
    predicted_trajs = original_list.reshape(num_of_trajs, predicted_trajs.shape[2], 2)  # 重新调整数组的形状 <class 'tuple'>: (20, 4, 2)

    # 0强度 1风速
    predicted_trajs[:, :, 0] = predicted_trajs[:, :, 0] * 50 + 960
    predicted_trajs[:, :, 1] = predicted_trajs[:, :, 1] * 25 + 40
    # 实际强度+风速 ground truth
    gt_traj_copy[:, :, 0] = gt_traj_copy[:, :, 0] * 50 + 960
    gt_traj_copy[:, :, 1] = gt_traj_copy[:, :, 1] * 25 + 40


    interpolation = predicted_trajs - gt_traj_copy #(6, 4, 2) 差值 (,T,)

    interpolation = torch.from_numpy(interpolation)


    loss_intensity = interpolation[:, :, 0] #torch.
    loss_wind = interpolation[:, :, 1]

    abs_inten = torch.abs(loss_intensity)
    abs_wind = torch.abs(loss_wind)

    min_value_intensity = torch.tensor([0.0, 0.0, 0.0, 0.0])
    min_value_wind = torch.tensor([0.0, 0.0, 0.0, 0.0])
    min_index_intensity = torch.tensor([0.0, 0.0, 0.0, 0.0])
    min_index_wind = torch.tensor([0.0, 0.0, 0.0, 0.0])

    for t in range(abs_inten.size(1)):
        min_value_intensity[t], min_index_intensity[t] = torch.min(abs_inten[:, t], dim=0)

    for t in range(abs_wind.size(1)):
        min_value_wind[t], min_index_wind[t] = torch.min(abs_wind[:, t], dim=0)

    sum_min_value_intensity = torch.sum(min_value_intensity, dim=-1)
    sum_min_value_wind = torch.sum(min_value_wind, dim=-1)

    return sum_min_value_intensity, sum_min_value_wind, min_index_intensity, min_index_wind, \
           min_value_intensity, min_value_wind,\
           predicted_trajs, gt_traj_copy #-,-,实际误差最小的一条轨迹的误差值（4）

def compute_ade_x_y_traj_aver(predicted_trajs, gt_traj):
    predicted_trajs = predicted_trajs[:, :, :, 0:2]
    gt_traj = gt_traj[:, 0:2]
    num_of_trajs = predicted_trajs.shape[1] #20条轨迹/6条轨迹
    # 将tuple1复制为一个列表，重复20次
    tuples = [gt_traj] * num_of_trajs
    gt_traj_copy = np.array(tuples) #<class 'tuple'>: (20, 4, 2)

    original_list = np.array(predicted_trajs)  # 将列表转换为NumPy数组
    predicted_trajs = original_list.reshape(num_of_trajs, predicted_trajs.shape[2], 2)  # 重新调整数组的形状 <class 'tuple'>: (20, 4, 2)

    # 0经度  1纬度  实际经纬度 预测值
    predicted_trajs[:, :, 0] = predicted_trajs[:, :, 0] / 10 * 500 + 1800
    predicted_trajs[:, :, 1] = predicted_trajs[:, :, 1] / 6 * 300
    #实际经纬度 ground truth
    gt_traj_copy[:, :, 0] = gt_traj_copy[:, :, 0] / 10 * 500 + 1800
    gt_traj_copy[:, :, 1] = gt_traj_copy[:, :, 1] / 6 * 300

    interpolation = predicted_trajs - gt_traj_copy #(20, 4, 2) 差值
    # 实际的距离上的差值
    interpolation[:, :, 0] = (interpolation[:, :, 0] / 10) * 111
    interpolation[:, :, 1] = (interpolation[:, :, 1] / 10) * 111 * np.cos(gt_traj_copy[:, :, 1] / 10 * np.pi / 180)
    #
    interpolation = torch.from_numpy(interpolation) #torch.Size([20条数, 4, 2经纬度])

    interpolation = interpolation ** 2

    loss = torch.sqrt(interpolation[:,:,0]+interpolation[:,:,1]) #torch.Size([6, 4个时刻])

    min_value_of_4_time = torch.mean(loss,dim=0)
    min_index = torch.tensor([0, 0, 0, 0])
    # for t in range(loss.size(1)):
    #     min_value_of_4_time[t], min_index[t] = torch.min(loss[:, t], dim=0)

    sum_min_value = torch.sum(min_value_of_4_time, dim=-1)

    return sum_min_value, min_index, min_value_of_4_time, predicted_trajs, gt_traj_copy #-,-,实际误差最小的一条轨迹的误差值（4）

# <class 'tuple'>: (1, 6, 4, 4)
def compute_ade_x_y_intensity_aver(predicted_trajs, gt_traj):
    predicted_trajs = predicted_trajs[:,:,:,2:4]
    gt_traj = gt_traj[:, 2:4]

    num_of_trajs = predicted_trajs.shape[1] #20条轨迹/6条轨迹
    # 将tuple1复制为一个列表，重复20次
    tuples = [gt_traj] * num_of_trajs
    gt_traj_copy = np.array(tuples) #<class 'tuple'>: (20, 4, 2)

    original_list = np.array(predicted_trajs)  # 将列表转换为NumPy数组
    predicted_trajs = original_list.reshape(num_of_trajs, predicted_trajs.shape[2], 2)  # 重新调整数组的形状 <class 'tuple'>: (20, 4, 2)

    # 0强度 1风速
    predicted_trajs[:, :, 0] = predicted_trajs[:, :, 0] * 50 + 960
    predicted_trajs[:, :, 1] = predicted_trajs[:, :, 1] * 25 + 40
    # 实际强度+风速 ground truth
    gt_traj_copy[:, :, 0] = gt_traj_copy[:, :, 0] * 50 + 960
    gt_traj_copy[:, :, 1] = gt_traj_copy[:, :, 1] * 25 + 40


    interpolation = predicted_trajs - gt_traj_copy #(6, 4, 2) 差值 (,T,)

    interpolation = torch.from_numpy(interpolation)


    loss_intensity = interpolation[:, :, 0] #torch.
    loss_wind = interpolation[:, :, 1]

    abs_inten = torch.abs(loss_intensity)
    abs_wind = torch.abs(loss_wind)

    min_value_intensity = torch.mean(abs_inten,dim=0)
    min_value_wind = torch.mean(abs_wind,dim=0)
    min_index_intensity = torch.tensor([0.0, 0.0, 0.0, 0.0])
    min_index_wind = torch.tensor([0.0, 0.0, 0.0, 0.0])

    # for t in range(abs_inten.size(1)):
    #     min_value_intensity[t], min_index_intensity[t] = torch.min(abs_inten[:, t], dim=0)
    #
    # for t in range(abs_wind.size(1)):
    #     min_value_wind[t], min_index_wind[t] = torch.min(abs_wind[:, t], dim=0)

    sum_min_value_intensity = torch.sum(min_value_intensity, dim=-1)
    sum_min_value_wind = torch.sum(min_value_wind, dim=-1)

    return sum_min_value_intensity, sum_min_value_wind, min_index_intensity, min_index_wind, \
           min_value_intensity, min_value_wind,\
           predicted_trajs, gt_traj_copy #-,-,实际误差最小的一条轨迹的误差值（4）


def compute_fde(predicted_trajs, gt_traj):
    final_error = np.linalg.norm(predicted_trajs[:, :, -1] - gt_traj[-1], axis=-1) ##<class 'tuple'>: (1, 20, 2) <class 'tuple'>: (2,)
    return final_error.flatten()


def compute_kde_nll(predicted_trajs, gt_traj):
    kde_ll = 0.
    log_pdf_lower_bound = -20
    num_timesteps = gt_traj.shape[0]
    num_batches = predicted_trajs.shape[0]

    for batch_num in range(num_batches):
        for timestep in range(num_timesteps):
            try:
                kde = gaussian_kde(predicted_trajs[batch_num, :, timestep].T)
                pdf = np.clip(kde.logpdf(gt_traj[timestep].T), a_min=log_pdf_lower_bound, a_max=None)[0]
                kde_ll += pdf / (num_timesteps * num_batches)
            except np.linalg.LinAlgError:
                kde_ll = np.nan

    return -kde_ll


def compute_obs_violations(predicted_trajs, map):
    obs_map = map.data

    interp_obs_map = RectBivariateSpline(range(obs_map.shape[1]),
                                         range(obs_map.shape[0]),
                                         binary_dilation(obs_map.T, iterations=4),
                                         kx=1, ky=1)

    old_shape = predicted_trajs.shape
    pred_trajs_map = map.to_map_points(predicted_trajs.reshape((-1, 2)))

    traj_obs_values = interp_obs_map(pred_trajs_map[:, 0], pred_trajs_map[:, 1], grid=False)
    traj_obs_values = traj_obs_values.reshape((old_shape[0], old_shape[1]))
    num_viol_trajs = np.sum(traj_obs_values.max(axis=1) > 0, dtype=float)

    return num_viol_trajs

# 算误差
def compute_batch_statistics(prediction_output_dict, #预测的轨迹结果
                             dt,
                             max_hl,
                             ph,
                             node_type_enum,
                             kde=True,
                             obs=False,
                             map=None,
                             prune_ph_to_future=False,
                             best_of=False): #True

# prediction_output_to_trajectories 函数用于将预测输出字典 (prediction_output_dict) 转换为轨迹。
# 它接受时间步长 (dt)、最大历史长度 (max_hl)、预测范围 (ph)
# 和 prune_ph_to_future 标志作为参数，用于控制将预测范围修剪到未来。
    #轨迹：预测轨迹+真实轨迹
    (prediction_dict,
     _,
     futures_dict) = prediction_output_to_trajectories(prediction_output_dict,
                                                       dt, #时间步长 (dt) ==多少？
                                                       max_hl, #最大历史长度 (max_hl) 应该8
                                                       ph, #预测范围 (ph)  ==多少？ 应该是4
                                                       prune_ph_to_future=prune_ph_to_future)
#prediction_output_dict：一个list 3组 <class 'tuple'>: (1, 6, 4, 4)
# prediction_dict：一个list 3组 <class 'tuple'>: (1, 6, 4, 4)
# futures_dict： 一个list 3组<class 'tuple'>: (4, 4)

    batch_error_dict = dict()
    for node_type in node_type_enum:
        batch_error_dict[node_type] = {'ade': list(), 'fde': list(),
                                       'distance': list(),
                                       'inten_di': list(), 'wind_di': list(),
                                       'real_dev': list(),
                                       'real_dev_intensity': list(),'real_dev_wind': list(),
                                       'predicted_trajs': list(),
                                       'predicted_inten_wind': list(),
                                       'gt_trajs': list(),
                                       'gt_inten_wind': list(),
                                       'kde': list(), 'obs_viols': list()}

#代码遍历预测字典 (prediction_dict) 的每个时间步 (t)，以及该时间步内的每个节点。一个时间步里有多个节点？？
# 它计算预测轨迹 (prediction_dict[t][node]) 与真实轨迹 (futures_dict[t][node])
# 之间的 ADE (平均位移误差) 和 FDE (最终位移误差)。
    for t in prediction_dict.keys(): #[7,8,9]
        for node in prediction_dict[t].keys(): #t=7/8 一个节点
            #pdb.set_trace()
            #target_shape =          <class 'tuple'>: (1, 20, 4, 2) <class 'tuple'>: (4, 2)
            #torch.Size([]) torch.Size([]) torch.Size([4])
            #                min_index, loss[min_index.item(),:]
            min_of_20_value, min_index, real_dev_distance, \
            predicted_trajs, gt_traj_copy = compute_ade_x_y_traj_each_time(prediction_dict[t][node], futures_dict[t][node]) #计算sample得到的prediction_dict与futures_dict之间的误差
            # compute_ade_x_y_traj_aver  compute_ade_x_y_traj_each_time

            min_of_20_value_intensity, min_of_20_value_wind,\
            min_index_intensity, min_index_wind, \
            real_dev_distance_intensity, real_dev_distance_wind,\
            predicted_inten_wind, gt_inten_wind_copy = compute_ade_x_y_intensity_each_time(
                prediction_dict[t][node], futures_dict[t][node])  # 计算sample得到的prediction_dict与futures_dict之间的误差
            #compute_ade_x_y_intensity_aver  compute_ade_x_y_intensity_each_time
            gt_trajs = gt_traj_copy[0]
            # predicted_trajs:预测轨迹（6，4，2）
            # gt_traj_copy：实际轨迹（6，4，2）只取第一组 后面5组是一样的
            # 缺失前8个点的坐标值
            gt_inten_wind = gt_inten_wind_copy[0] #（4，2）
            ade_errors = compute_ade(prediction_dict[t][node], futures_dict[t][node]) #<class 'tuple'>: (20,)
            fde_errors = compute_fde(prediction_dict[t][node], futures_dict[t][node])
            if kde:
                kde_ll = compute_kde_nll(prediction_dict[t][node], futures_dict[t][node])
            else:
                kde_ll = 0
            if obs:
                obs_viols = compute_obs_violations(prediction_dict[t][node], map)
            else:
                obs_viols = 0
            if best_of:
                ade_errors = np.min(ade_errors, keepdims=True) #ade_errors原始为20个点与未来轨迹的误差 取最小值后 指的是与实际轨迹最接近的那条轨迹
                fde_errors = np.min(fde_errors, keepdims=True)
                kde_ll = np.min(kde_ll)
            batch_error_dict[node.type]['ade'].extend(list(ade_errors))
            batch_error_dict[node.type]['fde'].extend(list(fde_errors))

            batch_error_dict[node.type]['distance'].extend([min_of_20_value])

            batch_error_dict[node.type]['inten_di'].extend([min_of_20_value_intensity])
            batch_error_dict[node.type]['wind_di'].extend([min_of_20_value_wind])

            real_dev_distance_intensity = torch.abs(real_dev_distance_intensity)
            real_dev_distance_wind = torch.abs(real_dev_distance_wind)

            batch_error_dict[node.type]['real_dev'].append(real_dev_distance.tolist())

            batch_error_dict[node.type]['real_dev_intensity'].append(real_dev_distance_intensity.tolist())
            batch_error_dict[node.type]['real_dev_wind'].append(real_dev_distance_wind.tolist())

            batch_error_dict[node.type]['predicted_trajs'].append(list(predicted_trajs))

            batch_error_dict[node.type]['predicted_inten_wind'].append(list(predicted_inten_wind))

            batch_error_dict[node.type]['gt_trajs'].append(list(gt_trajs))
            batch_error_dict[node.type]['gt_inten_wind'].append(list(gt_inten_wind))
            batch_error_dict[node.type]['kde'].extend([kde_ll])
            batch_error_dict[node.type]['obs_viols'].extend([obs_viols])

    return batch_error_dict


# def log_batch_errors(batch_errors_list, log_writer, namespace, curr_iter, bar_plot=[], box_plot=[]):
#     for node_type in batch_errors_list[0].keys():
#         for metric in batch_errors_list[0][node_type].keys():
#             metric_batch_error = []
#             for batch_errors in batch_errors_list:
#                 metric_batch_error.extend(batch_errors[node_type][metric])

#             if len(metric_batch_error) > 0:
#                 log_writer.add_histogram(f"{node_type.name}/{namespace}/{metric}", metric_batch_error, curr_iter)
#                 log_writer.add_scalar(f"{node_type.name}/{namespace}/{metric}_mean", np.mean(metric_batch_error), curr_iter)
#                 log_writer.add_scalar(f"{node_type.name}/{namespace}/{metric}_median", np.median(metric_batch_error), curr_iter)

#                 if metric in bar_plot:
#                     pd = {'dataset': [namespace] * len(metric_batch_error),
#                                   metric: metric_batch_error}
#                     kde_barplot_fig, ax = plt.subplots(figsize=(5, 5))
#                     visualization.visualization_utils.plot_barplots(ax, pd, 'dataset', metric)
#                     log_writer.add_figure(f"{node_type.name}/{namespace}/{metric}_bar_plot", kde_barplot_fig, curr_iter)

#                 if metric in box_plot:
#                     mse_fde_pd = {'dataset': [namespace] * len(metric_batch_error),
#                                   metric: metric_batch_error}
#                     fig, ax = plt.subplots(figsize=(5, 5))
#                     visualization.visualization_utils.plot_boxplots(ax, mse_fde_pd, 'dataset', metric)
#                     log_writer.add_figure(f"{node_type.name}/{namespace}/{metric}_box_plot", fig, curr_iter)


def print_batch_errors(batch_errors_list, namespace, curr_iter):
    for node_type in batch_errors_list[0].keys():
        for metric in batch_errors_list[0][node_type].keys():
            metric_batch_error = []
            for batch_errors in batch_errors_list:
                metric_batch_error.extend(batch_errors[node_type][metric])

            if len(metric_batch_error) > 0:
                print(f"{curr_iter}: {node_type.name}/{namespace}/{metric}_mean", np.mean(metric_batch_error))
                print(f"{curr_iter}: {node_type.name}/{namespace}/{metric}_median", np.median(metric_batch_error))


def batch_pcmd(prediction_output_dict,
               dt,
               max_hl,
               ph,
               node_type_enum,
               kde=True,
               obs=False,
               map=None,
               prune_ph_to_future=False,
               best_of=False):

    (prediction_dict,
     _,
     futures_dict) = prediction_output_to_trajectories(prediction_output_dict,
                                                       dt,
                                                       max_hl,
                                                       ph,
                                                       prune_ph_to_future=prune_ph_to_future)

    batch_error_dict = dict()
    for node_type in node_type_enum:
        batch_error_dict[node_type] =  {'ade': list(), 'fde': list(), 'kde': list(), 'obs_viols': list()}

    for t in prediction_dict.keys():
        for node in prediction_dict[t].keys():
            ade_errors = compute_ade(prediction_dict[t][node], futures_dict[t][node])
            fde_errors = compute_fde(prediction_dict[t][node], futures_dict[t][node])
            if kde:
                kde_ll = compute_kde_nll(prediction_dict[t][node], futures_dict[t][node])
            else:
                kde_ll = 0
            if obs:
                obs_viols = compute_obs_violations(prediction_dict[t][node], map)
            else:
                obs_viols = 0
            if best_of:
                ade_errors = np.min(ade_errors, keepdims=True)
                fde_errors = np.min(fde_errors, keepdims=True)
                kde_ll = np.min(kde_ll)
            batch_error_dict[node.type]['ade'].append(np.array(ade_errors))
            batch_error_dict[node.type]['fde'].append(np.array(fde_errors))

    return batch_error_dict
