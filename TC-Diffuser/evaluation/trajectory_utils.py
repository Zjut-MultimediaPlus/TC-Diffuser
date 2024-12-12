import numpy as np
import pdb

def prediction_output_to_trajectories(prediction_output_dict,
                                      dt,
                                      max_h,
                                      ph,
                                      map=None,
                                      prune_ph_to_future=False):

    prediction_timesteps = prediction_output_dict.keys() #预测的时刻 预测第9时刻 然后前8 后4 应该是一个数组 从9，10.。。

    output_dict = dict()#存储输出轨迹、历史轨迹和未来轨迹
    histories_dict = dict()
    futures_dict = dict()

    for t in prediction_timesteps: #预测的时刻 一个个看过去 t应该取到往后取 长度不够了为止 预测第9时刻 然后前8 后4
        histories_dict[t] = dict()
        output_dict[t] = dict()
        futures_dict[t] = dict()
        prediction_nodes = prediction_output_dict[t].keys()
        for node in prediction_nodes: #遍历预测节点列表prediction_nodes node：PEDESTRIAN/2
            predictions_output = prediction_output_dict[t][node] #获取特定时间步和节点的预测输出 t=9,node：PEDESTRIAN/2
            position_state = {'position': ['x', 'y'],'intensity': ['x', 'y']}
            #从节点对象中获取时间步t-max_h到t之间的历史轨迹history，包括当前位置。如果存在NaN值，将其从轨迹中删除
            history = node.get(np.array([t - max_h, t]), position_state)  # History includes current pos 8个坐标 过去的八个时刻 一个节点
            history = history[~np.isnan(history.sum(axis=1))]
            #pdb.set_trace()
            #从节点对象中获取时间步t+1到t+ph之间的未来轨迹future。如果存在NaN值，将其从轨迹中删除。
            future = node.get(np.array([t + 1, t + ph]), position_state) #future就是真实的轨迹 ph应该取4
            # 一个node 只取一次 但在时刻中 每个时刻都取
            # replace nan to 0
            #future[np.isnan(future)] = 0
            future = future[~np.isnan(future.sum(axis=1))]

            if prune_ph_to_future: #如果prune_ph_to_future为真，
                # 则将预测输出predictions_output裁剪为与未来轨迹长度相同的部分。
                # 如果裁剪后的预测输出长度为0，则继续下一个节点的处理
                predictions_output = predictions_output[:, :, :future.shape[0]]
                if predictions_output.shape[2] == 0:
                    continue

            trajectory = predictions_output #将预测输出predictions_output作为轨迹保存在trajectory中

            if map is None:
                histories_dict[t][node] = history #将原始历史、输出和未来轨迹保存在相应的字典中
                output_dict[t][node] = trajectory
                futures_dict[t][node] = future
            else:
                histories_dict[t][node] = map.to_map_points(history) #将历史、输出和未来轨迹转换为地图坐标系中的点
                output_dict[t][node] = map.to_map_points(trajectory)
                futures_dict[t][node] = map.to_map_points(future) #真值

    return output_dict, histories_dict, futures_dict #比较output_dict=predict和future
#将预测输出转换为轨迹，并根据提供的地图将轨迹转换为地图坐标系中的点。这样的转换有助于后续的轨迹评估和可视化等任务。