from torch.utils import data
import numpy as np
from .preprocessing import get_node_timestep_data

# self.train_dataset = EnvironmentDataset(train_env,
#                                         self.hyperparams['state'],
#                                         self.hyperparams['pred_state'],  # 需要预测的状态是 两个速度
#                                         scene_freq_mult=self.hyperparams['scene_freq_mult_train'],  # none
#                                         node_freq_mult=self.hyperparams['node_freq_mult_train'],  # none
#                                         hyperparams=self.hyperparams,
#                                         min_history_timesteps=1,
#                                         min_future_timesteps=self.hyperparams['prediction_horizon'],  # 12 预测12步--4
#                                         return_robot=not self.config.incl_robot_node)
class EnvironmentDataset(object):
    def __init__(self, env, state, pred_state, node_freq_mult, scene_freq_mult, hyperparams, **kwargs):
        self.env = env
        self.state = state
        self.pred_state = pred_state
        self.hyperparams = hyperparams
        self.max_ht = self.hyperparams['maximum_history_length']
        self.max_ft = kwargs['min_future_timesteps']
        self.node_type_datasets = list()  #外面取的node_type_data_set,是NodeTypeDataset组成的list
        self._augment = False
        for node_type in env.NodeType:
            if node_type not in hyperparams['pred_state']:
                continue
            self.node_type_datasets.append(NodeTypeDataset(env, node_type, state, pred_state, node_freq_mult,
                                                           scene_freq_mult, hyperparams, **kwargs))

    @property
    def augment(self): #是否进行数据增强
        return self._augment

    @augment.setter
    def augment(self, value):
        self._augment = value
        for node_type_dataset in self.node_type_datasets:
            node_type_dataset.augment = value

    def __iter__(self):
        return iter(self.node_type_datasets)


class NodeTypeDataset(data.Dataset):
    def __init__(self, env, node_type, state, pred_state, node_freq_mult,
                 scene_freq_mult, hyperparams, augment=False, **kwargs):
        self.env = env
        self.state = state
        self.pred_state = pred_state
        self.hyperparams = hyperparams
        self.max_ht = self.hyperparams['maximum_history_length']
        self.max_ft = kwargs['min_future_timesteps']

        self.augment = augment

        self.node_type = node_type
        self.index = self.index_env(node_freq_mult, scene_freq_mult, **kwargs)
        self.len = len(self.index)
        self.edge_types = [edge_type for edge_type in env.get_edge_types() if edge_type[0] is node_type]

    # index_env(self, node_freq_mult, scene_freq_mult, **kwargs): 根据节点频率倍数和场景频率倍数，
    # 在环境中索引所有存在的特定节点，并生成索引列表。
    # 对于每个场景和时间步，它将特定节点添加到索引列表中，重复的次数由节点和场景的频率倍数决定。
    def index_env(self, node_freq_mult, scene_freq_mult, **kwargs):
        index = list()
        for scene in self.env.scenes:
            # 有可能是这里的 时刻导致样本少？
            present_node_dict = scene.present_nodes(np.arange(0, scene.timesteps), type=self.node_type, **kwargs)
            for t, nodes in present_node_dict.items():
                for node in nodes:
                    if((t-self.max_ht)>=0):
                        index += [(scene, t, node)] *\
                             (scene.frequency_multiplier if scene_freq_mult else 1) *\
                             (node.frequency_multiplier if node_freq_mult else 1)

        return index

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        (scene, t, node) = self.index[i]

        if self.augment:
            scene = scene.augment()
            node = scene.get_node_by_id(node.id)
        #
        return get_node_timestep_data(self.env, scene, t, node, self.state, self.pred_state,
                                      self.edge_types, self.max_ht, self.max_ft, self.hyperparams)
