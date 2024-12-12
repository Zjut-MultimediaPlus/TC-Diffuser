import torch
import numpy as np
import collections.abc
from torch.utils.data._utils.collate import default_collate
import dill
container_abcs = collections.abc

#restore 函数用于判断输入的数据是否是被序列化的字节流，
# 如果是，则将其还原为原始的数据结构；
# 如果不是，则直接返回原始的数据结构。
# 这样可以确保在多进程环境下，正确地还原共享的数据结构。
def restore(data):
    """
    In case we dilled some structures to share between multiple process this function will restore them.
    If the data input are not bytes we assume it was not dilled in the first place

    :param data: Possibly dilled data structure
    :return: Un-dilled data structure
    """
    if type(data) is bytes:
        return dill.loads(data)
    return data

# 上述代码定义了一个名为 `collate` 的函数，用于自定义数据集的批处理方式。

# 该函数接受一个 `batch` 参数，该参数是由数据集中取出的一批数据样本组成的列表。
# 函数首先检查 `batch` 的长度，如果长度为0，则直接返回 `batch`。
# 然后，判断第一个元素 `elem` 是否是一个可序列（`container_abcs.Sequence`）对象。
# 如果是，说明 `batch` 中的元素是可迭代的数据样本，可能是由多个数据组成的。
# 在这种情况下，函数会进行递归处理，将每个样本中的数据再次进行批处理。
# 如果 `elem` 是一个映射（`container_abcs.Mapping`）对象，
# 则说明 `batch` 中的元素是字典形式的数据样本，可能包含各种数据信息。在这种情况下，
# 函数会对字典中的值进行处理，并将结果以字典形式返回。
# 最后，如果 `elem` 不是可序列或映射对象，
# 说明 `batch` 中的元素是基本数据类型（如张量等），
# 则调用默认的 `default_collate` 函数进行批处理。
# 总的来说，`collate` 函数主要用于处理数据样本的批处理，
# 并根据不同的数据类型递归地对数据进行组合和处理。
# 这样可以确保在构建数据加载器时，
# 每个批次中的数据能够以合适的方式进行处理和组织，以满足模型的输入要求。
def collate(batch): #这里的batch已经定形了
    if len(batch) == 0:
        return batch
    elem = batch[0]
    if elem is None:
        return None
    elif isinstance(elem, container_abcs.Sequence):
        if len(elem) == 4: # We assume those are the maps, map points, headings and patch_size
            scene_map, scene_pts, heading_angle, patch_size = zip(*batch)
            if heading_angle[0] is None:
                heading_angle = None
            else:
                heading_angle = torch.Tensor(heading_angle)
            map = scene_map[0].get_cropped_maps_from_scene_map_batch(scene_map,
                                                                     scene_pts=torch.Tensor(scene_pts),
                                                                     patch_size=patch_size[0],
                                                                     rotation=heading_angle)
            return map
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]
    elif isinstance(elem, container_abcs.Mapping):
        # We have to dill the neighbors structures. Otherwise each tensor is put into
        # shared memory separately -> slow, file pointer overhead
        # we only do this in multiprocessing
        neighbor_dict = {key: [d[key] for d in batch] for key in elem}
        return dill.dumps(neighbor_dict) if torch.utils.data.get_worker_info() else neighbor_dict
    return default_collate(batch)


def get_relative_robot_traj(env, state, node_traj, robot_traj, node_type, robot_type):
    # TODO: We will have to make this more generic if robot_type != node_type
    # Make Robot State relative to node
    _, std = env.get_standardize_params(state[robot_type], node_type=robot_type)
    std[0:2] = env.attention_radius[(node_type, robot_type)]
    robot_traj_st = env.standardize(robot_traj,
                                    state[robot_type],
                                    node_type=robot_type,
                                    mean=node_traj,
                                    std=std)
    robot_traj_st_t = torch.tensor(robot_traj_st, dtype=torch.float)

    return robot_traj_st_t

# 对给定时间步的单个节点进行数据预处理，包括节点状态随时间的变化和相邻节点数据。
def get_node_timestep_data(env, scene, t, node, state, pred_state,
                           edge_types, max_ht, max_ft, hyperparams,
                           scene_graph=None):
    """
    Pre-processes the data for a single batch element: node state over time for a specific time in a specific scene
    as well as the neighbour data for it.

    :param env: Environment
    :param scene: Scene
    :param t: Timestep in scene
    :param node: Node
    :param state: Specification of the node state
    :param pred_state: Specification of the prediction state
    :param edge_types: List of all Edge Types for which neighbours are pre-processed
    :param max_ht: Maximum history timesteps
    :param max_ft: Maximum future timesteps (prediction horizon)
    :param hyperparams: Model hyperparameters
    :param scene_graph: If scene graph was already computed for this scene and time you can pass it here
    :return: Batch Element
    """

    # Node
    timestep_range_x = np.array([t - max_ht, t])
    timestep_range_y = np.array([t + 1, t + max_ft])
    #使用 node.get 方法获取节点在时间步 t 前后的状态。
    # 其中，timestep_range_x 表示历史时间步范围，timestep_range_y 表示未来时间步范围
    x = node.get(timestep_range_x, state[node.type]) #x纯粹是node。data的timestep_range_x范围的所有数据
    y = node.get(timestep_range_y, pred_state[node.type]) #y是node 的timestep_range_y范围内的 该预测的四个参数的原始值 也就是gt
    first_history_index = (max_ht - node.history_points_at(t)).clip(0)

    _, std = env.get_standardize_params(state[node.type], node.type)
    std[0:2] = env.attention_radius[(node.type, node.type)]
    std[6:8] = env.attention_radius[(node.type, node.type)]

    rel_state = np.zeros_like(x[0]) #全0
    rel_state[0:2] = np.array(x)[-1, 0:2] #x的最后一行 array[7] 最靠近当前的历史时刻（上一个时刻）
    rel_state[6:8] = np.array(x)[-1, 6:8]
    x_st = env.standardize(x, state[node.type], node.type, mean=rel_state, std=std)
    if list(pred_state[node.type].keys())[0] == 'position':  # If we predict position we do it relative to current pos
        y_st = env.standardize(y, pred_state[node.type], node.type, mean=rel_state[0:2]) #这块没跑到
    else:
        y_st = env.standardize(y, pred_state[node.type], node.type)

    x_t = torch.tensor(x, dtype=torch.float)
    y_t = torch.tensor(y, dtype=torch.float)
    x_st_t = torch.tensor(x_st, dtype=torch.float)
    y_st_t = torch.tensor(y_st, dtype=torch.float)

    # print("timestep_range_x:", timestep_range_x)
    # print("timestep_range_y:", timestep_range_y)
    gph_data_x = node.get(timestep_range_x, 'gph_data')
    gph_data_y = node.get(timestep_range_y, 'gph_data')

    # gph_data_min = 44490.578125
    # gph_data_max = 58768.4486860389

    # gph_data_x_norm = (gph_data_x - gph_data_min) / (gph_data_max - gph_data_min)
    # gph_data_y_norm = (gph_data_y - gph_data_min) / (gph_data_max - gph_data_min)

    gph_data_x_norm = gph_data_x
    gph_data_y_norm = gph_data_y

    rel_state_gph = gph_data_x_norm[-1]  # 全0
    mean_gph = rel_state_gph
    std_gph = np.zeros_like(gph_data_x_norm[0])
    std_gph = std_gph + 3 #原为3
    gph_data_x_norm_t = np.where(np.isnan(gph_data_x_norm), np.array(np.nan), (gph_data_x_norm - mean_gph) / std_gph) #-90，90
    gph_data_x_norm_t = gph_data_x_norm_t*0.01

    env_data_x = node.get(timestep_range_x, 'env_data')
    env_data_y = node.get(timestep_range_y, 'env_data')

    # Neighbors
    neighbors_data_st = None
    neighbors_edge_value = None
    if hyperparams['edge_encoding']: #真
        # Scene Graph
        # 通过调用 scene.get_scene_graph 方法获取当前时间步的场景图。场景图是描述节点之间连接关系的数据结构。
        scene_graph = scene.get_scene_graph(t,
                                            env.attention_radius,
                                            hyperparams['edge_addition_filter'],
                                            hyperparams['edge_removal_filter']) if scene_graph is None else scene_graph
        #neighbors_data_st 是一个字典，
        # 用于存储不同类型的边对应的邻居节点数据。
        # 每个边类型都对应一个空列表
        neighbors_data_st = dict()
        #：neighbors_edge_value 是一个字典，用于存储不同类型的边对应的邻居边值数据。
        neighbors_edge_value = dict()
        for edge_type in edge_types:
            neighbors_data_st[edge_type] = list()
            # We get all nodes which are connected to the current node for the current timestep
            connected_nodes = scene_graph.get_neighbors(node, edge_type[1])
            #确实yes
            if hyperparams['dynamic_edges'] == 'yes':
                # We get the edge masks for the current node at the current timestep
                edge_masks = torch.tensor(scene_graph.get_edge_scaling(node), dtype=torch.float)
                neighbors_edge_value[edge_type] = edge_masks

            for connected_node in connected_nodes:
                #处理邻居节点状态：对于每个邻居节点，
                # 使用 connected_node.get 方法获取其在历史时间步（t - max_ht 到 t）上的状态。
                # 然后，将邻居节点的状态与当前节点进行标准化，
                # 使它们具有相同的标准化参数。标准化参数通过 env.get_standardize_params 方法获取
                neighbor_state_np = connected_node.get(np.array([t - max_ht, t]),
                                                       state[connected_node.type],
                                                       padding=0.0)

                # Make State relative to node where neighbor and node have same state
                _, std = env.get_standardize_params(state[connected_node.type], node_type=connected_node.type)
                std[0:2] = env.attention_radius[edge_type]
                equal_dims = np.min((neighbor_state_np.shape[-1], x.shape[-1]))
                rel_state = np.zeros_like(neighbor_state_np)
                rel_state[:, ..., :equal_dims] = x[-1, ..., :equal_dims]
                neighbor_state_np_st = env.standardize(neighbor_state_np,
                                                       state[connected_node.type],
                                                       node_type=connected_node.type,
                                                       mean=rel_state,
                                                       std=std)

                neighbor_state = torch.tensor(neighbor_state_np_st, dtype=torch.float)
                neighbors_data_st[edge_type].append(neighbor_state)
                #neighbors_data_st 字典将包含与当前节点相连接的所有邻居节点的标准化状态数据，
                # 按照对应的边类型分别存储在各自的列表中。
                # neighbors_edge_value 字典将包含邻居边值数据

    # Robot
    robot_traj_st_t = None
    timestep_range_r = np.array([t, t + max_ft])
    ##假
    if hyperparams['incl_robot_node']:
        x_node = node.get(timestep_range_r, state[node.type])
        if scene.non_aug_scene is not None:
            robot = scene.get_node_by_id(scene.non_aug_scene.robot.id)
        else:
            robot = scene.robot
        robot_type = robot.type
        robot_traj = robot.get(timestep_range_r, state[robot_type], padding=0.0)
        robot_traj_st_t = get_relative_robot_traj(env, state, x_node, robot_traj, node.type, robot_type)

    # Map 假
    map_tuple = None
    if hyperparams['use_map_encoding']:
        if node.type in hyperparams['map_encoder']:
            if node.non_aug_node is not None:
                x = node.non_aug_node.get(np.array([t]), state[node.type])
            me_hyp = hyperparams['map_encoder'][node.type]
            if 'heading_state_index' in me_hyp:
                heading_state_index = me_hyp['heading_state_index']
                # We have to rotate the map in the opposit direction of the agent to match them
                if type(heading_state_index) is list:  # infer from velocity or heading vector
                    heading_angle = -np.arctan2(x[-1, heading_state_index[1]],
                                                x[-1, heading_state_index[0]]) * 180 / np.pi
                else:
                    heading_angle = -x[-1, heading_state_index] * 180 / np.pi
            else:
                heading_angle = None

            scene_map = scene.map[node.type]
            map_point = x[-1, :2]


            patch_size = hyperparams['map_encoder'][node.type]['patch_size']
            map_tuple = (scene_map, map_point, heading_angle, patch_size)

    return (first_history_index, x_t, y_t, x_st_t, y_st_t, neighbors_data_st,
            neighbors_edge_value, robot_traj_st_t, map_tuple
            , gph_data_x_norm, gph_data_y_norm
            , gph_data_x_norm_t
            , env_data_x, env_data_y
            , timestep_range_x,timestep_range_y
            )


def get_timesteps_data(env, scene, t, node_type, state, pred_state,
                       edge_types, min_ht, max_ht, min_ft, max_ft, hyperparams):
    """
    Puts together the inputs for ALL nodes in a given scene and timestep in it.

    :param env: Environment
    :param scene: Scene
    :param t: Timestep in scene
    :param node_type: Node Type of nodes for which the data shall be pre-processed
    :param state: Specification of the node state
    :param pred_state: Specification of the prediction state
    :param edge_types: List of all Edge Types for which neighbors are pre-processed
    :param max_ht: Maximum history timesteps #不管min?
    :param max_ft: Maximum future timesteps (prediction horizon)
    :param hyperparams: Model hyperparameters
    :return:
    """
    # 列表nodes_per_ts包含了所有满足条件的节点，然后可以对这些节点进行进一步的操作和处理。
    # tuple(22,6) 3个
    # 使用 scene.present_nodes 方法获取在给定时间步 t 中满足条件的所有节点。
    # 这些节点将保存在 nodes_per_ts 字典中，其中时间步作为键，对应的节点列表作为值
    nodes_per_ts = scene.present_nodes(t,
                                       type=node_type,
                                       min_history_timesteps=min_ht,#7
                                       min_future_timesteps=max_ft,#12--4
                                       return_robot=not hyperparams['incl_robot_node'])
    batch = list()
    nodes = list()
    out_timesteps = list()
    for timestep in nodes_per_ts.keys():
            scene_graph = scene.get_scene_graph(timestep,
                                                env.attention_radius,
                                                hyperparams['edge_addition_filter'],
                                                hyperparams['edge_removal_filter'])
            present_nodes = nodes_per_ts[timestep]
            #将获取的节点数据添加到 batch 列表中，
            # 并将节点 ID 和时间步分别添加到 nodes 和 out_timesteps 列表中。
            for node in present_nodes:
                if((timestep-max_ht)>=0):

                    data = get_node_timestep_data(env, scene, timestep, node, state, pred_state,
                                                    edge_types, max_ht, max_ft, hyperparams,
                                                    scene_graph=scene_graph)
                    batch.append(data)  # 将数据添加到列表中
                    nodes.append(node)
                    out_timesteps.append(timestep)

    #如果没有满足条件的节点数据，返回 None。
    if len(out_timesteps) == 0:
        return None
    #将 batch 列表中的节点数据使用 collate 函数进行组合，并返回组合后的张量、节点 ID 列表和时间步列表。
    return collate(batch), nodes, out_timesteps