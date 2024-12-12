def get_traj_hypers(): #轨迹的超参数
    hypers = {   'batch_size': 256,
    'grad_clip': 1.0,
    'learning_rate_style': 'exp',
    'min_learning_rate': 1e-05,
    'learning_decay_rate': 0.9999,
    'prediction_horizon': 4, #4  预测的时间步数
    'minimum_history_length': 1, #历史轨迹的最小和最大长度。
    'maximum_history_length': 7,
    'map_encoder': #地图编码器
        {'PEDESTRIAN':
            {'heading_state_index': 6, #行人状态中指示朝向的索引
             'patch_size': [50, 10, 50, 90], #地图编码器使用的补丁（patch）大小，指定了地图的区域范围
             'map_channels': 3, #地图数据的通道数
             'hidden_channels': [10, 20, 10, 1], #隐藏层通道数的列表，指定了地图编码器中的隐藏层的通道数
             'output_size': 32, #输出的特征向量大小
             'masks': [5, 5, 5, 5], #补丁的掩码（mask）大小，用于控制地图编码器的感受野
             'strides': [1, 1, 1, 1], #补丁的步幅（stride），指定了在地图上滑动补丁的步长
             'dropout': 0.5 #随机失活（dropout）率，用于在地图编码器的训练过程中防止过拟合
            }
        },
    'k': 1, #k: 训练过程中每个时间步的采样次数。
    'k_eval': 25, #在评估过程中每个时间步的采样次数
    'kl_min': 0.07, #KL散度（Kullback-Leibler divergence）的最小值，用于控制模型的多样性
    'kl_weight': 100.0,#KL散度的权重，用于平衡KL散度项与重构损失项之间的重要性
    'kl_weight_start': 0,#KL散度的权重，用于平衡KL散度项与重构损失项之间的重要性
    'kl_decay_rate': 0.99995, # KL散度权重的衰减率，用于控制KL散度权重的减小速度。
    'kl_crossover': 400,  #KL散度权重衰减的交叉点，用于控制KL散度权重开始衰减的时间步
    'kl_sigmoid_divisor': 4, # KL散度权重衰减过程中的sigmoid函数除数
    'rnn_kwargs':
        {'dropout_keep_prob': 0.75},
    'MLP_dropout_keep_prob': 0.9, #MLP_dropout_keep_prob: 多层感知机（MLP）中的dropout保留概率，用于控制在训练过程中随机丢弃的神经元比例
    'enc_rnn_dim_edge': 128,  #编码器（encoder）中使用的循环神经网络（RNN）的维度
    'enc_rnn_dim_edge_influence': 128,
    'enc_rnn_dim_history': 128, # 编码器中历史轨迹信息的循环神经网络的维度
    'enc_rnn_dim_future': 128, # 编码器中未来轨迹信息的循环神经网络的维度
    'dec_rnn_dim': 128, #解码器（decoder）中循环神经网络的维度
    'q_z_xy_MLP_dims': None, #Q网络中MLP的维度列表，用于学习潜在变量z的条件分布
    'p_z_x_MLP_dims': 32, #P网络中MLP的维度列表，用于学习潜在变量z的先验分布。
    'GMM_components': 1, #GMM_components: 混合高斯模型（GMM）的成分数量
    'log_p_yt_xz_max': 6,
    'N': 1, #: 采样次数，表示从潜在变量空间中采样的次数。
    'tau_init': 2.0,
    'tau_final': 0.05,
    'tau_decay_rate': 0.997,
    'use_z_logit_clipping': True,
    'z_logit_clip_start': 0.05,
    'z_logit_clip_final': 5.0,
    'z_logit_clip_crossover': 300,
    'z_logit_clip_divisor': 5,
    'dynamic': #定义了动力学模型的类型和属性
        {'PEDESTRIAN':
            {'name': 'SingleIntegrator', #PEDESTRIAN 的动力学模型类型是 SingleIntegrator，表示采用单一积分器模型
             'distribution': False,
             'limits': {}
            }
        },
    'state': #定义了状态的属性
        {'PEDESTRIAN': #PEDESTRIAN 的状态包括位置 (position)、速度 (velocity) 和加速度 (acceleration)，并且每个属性都有 x 和 y 两个分量。
            {'position': ['x', 'y'],
             'velocity': ['x', 'y'],
             'acceleration': ['x', 'y']
                ,
             'intensity': ['x', 'y'],   #x:强度 y:风速
             'velocity_i': ['x', 'y'],
             'acceleration_i': ['x', 'y']
            }
        },
    'pred_state': {'PEDESTRIAN': {'velocity': ['x', 'y']
        ,'velocity_i': ['x', 'y']
                                  }}, #预测状态只包括速度 (velocity)，并且具体到每个分量的 x 和 y
    'log_histograms': False,
    'dynamic_edges': 'yes',
    'edge_state_combine_method': 'sum',
    'edge_influence_combine_method': 'attention',
    'edge_addition_filter': [0.25, 0.5, 0.75, 1.0],
    'edge_removal_filter': [1.0, 0.0],
    'offline_scene_graph': 'yes',
    'incl_robot_node': False,
    'node_freq_mult_train': False,
    'node_freq_mult_eval': False,
    'scene_freq_mult_train': False,
    'scene_freq_mult_eval': False,
    'scene_freq_mult_viz': False,
    'edge_encoding': True,
    'use_map_encoding': False,  #现在的轨迹是不使用map的 但写好的是有的
    'augment': True,
    'override_attention_radius': [],
    'learning_rate': 0.01,
    'npl_rate': 0.8,
    'K': 80, #每个迭代步骤从训练数据中随机选择 80 个样本进行训练
    'tao': 0.4} #定义了一个指数衰减的学习率策略 调整学习率
    return hypers
