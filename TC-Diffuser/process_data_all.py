import sys
import os
import numpy as np
import pandas as pd
import dill
import pickle

from environment import Environment, Scene, Node, derivative_of

desired_max_time = 100
pred_indices = [2, 3]
state_dim = 6
frame_diff = 10
desired_frame_diff = 1
dt = 0.4

standardization = {
    'PEDESTRIAN': {
        'position': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        },
        'velocity': {
            'x': {'mean': 0, 'std': 2},
            'y': {'mean': 0, 'std': 2}
        },
        'acceleration': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        }

        ,
        'intensity': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        },
        'velocity_i': {
            'x': {'mean': 0, 'std': 2},
            'y': {'mean': 0, 'std': 2}
        },
        'acceleration_i': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        }

        # ,
        # 'gph':{
        #     'x': {'mean': 0, 'std': 1,
        #             'mean': 0, 'std': 1
        #
        #
        #           },
        #     'y': {'mean': 0, 'std': 1}
        # }


    }
}

def maybe_makedirs(path_to_create):
    """This function will create a directory, unless it exists already,
    at which point the function will return.
    The exception handling is necessary as it prevents a race condition
    from occurring.
    Inputs:
        path_to_create - A string path to a directory you'd like created.
    """
    try:
        os.makedirs(path_to_create)
    except OSError:
        if not os.path.isdir(path_to_create):
            raise

def augment_scene(scene, angle):
    def rotate_pc(pc, alpha):
        M = np.array([[np.cos(alpha), -np.sin(alpha)],
                      [np.sin(alpha), np.cos(alpha)]])
        return M @ pc

    data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration',
                                                'intensity', 'velocity_i', 'acceleration_i'
                                                # ,'gph'
                                                ], ['x', 'y']])

    scene_aug = Scene(timesteps=scene.timesteps, dt=scene.dt, name=scene.name)

    alpha = angle * np.pi / 180

    for node in scene.nodes:
        x = node.data.position.x.copy()
        y = node.data.position.y.copy()

        x, y = rotate_pc(np.array([x, y]), alpha)

        vx = derivative_of(x, scene.dt)
        vy = derivative_of(y, scene.dt)
        ax = derivative_of(vx, scene.dt)
        ay = derivative_of(vy, scene.dt)

        #intensity
        x_i = node.data.intensity.x.copy()
        y_i = node.data.intensity.y.copy()

        x_i, y_i = rotate_pc(np.array([x_i, y_i]), alpha)

        vx_i = derivative_of(x_i, scene.dt)
        vy_i = derivative_of(y_i, scene.dt)
        ax_i = derivative_of(vx_i, scene.dt)
        ay_i = derivative_of(vy_i, scene.dt)

        gph = node.gph_data
        env_data = node.env_data


        data_dict = {('position', 'x'): x,
                     ('position', 'y'): y,
                     ('velocity', 'x'): vx,
                     ('velocity', 'y'): vy,
                     ('acceleration', 'x'): ax,
                     ('acceleration', 'y'): ay,

                     ('intensity', 'x'): x_i,
                     ('intensity', 'y'): y_i,
                     ('velocity_i', 'x'): vx_i,
                     ('velocity_i', 'y'): vy_i,
                     ('acceleration_i', 'x'): ax_i,
                     ('acceleration_i', 'y'): ay_i

                     # ,
                     # ('gph', 'x'): gph,
                     # ('gph', 'y'): gph1

                     }

        node_data = pd.DataFrame(data_dict, columns=data_columns)

        node = Node(node_type=node.type, node_id=node.id, data=node_data,
                    gph_data = gph,
                    env_data = env_data,
                    first_timestep=node.first_timestep)

        scene_aug.nodes.append(node)
    return scene_aug


def augment(scene):
    scene_aug = np.random.choice(scene.augmented)
    scene_aug.temporal_scene_graph = scene.temporal_scene_graph
    return scene_aug


nl = 0
l = 0

data_folder_name = 'processed_data_noise_traj_inten_wind_gph_env'

maybe_makedirs(data_folder_name)
# data_columns这个多级索引对象被用于表示一个包含位置、速度和加速度的数据集，其中每个变量都有x和y两个维度
# 使用这个多级索引对象data_columns可以创建一个包含位置、速度和加速度数据的DataFrame，其中每个变量都有x和y两个维度的取值
data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration',
                                            'intensity','velocity_i', 'acceleration_i'
                                            # ,'gph'
                                            ], ['x', 'y']])

# Process ETH-UCY
#        五个不同的场景： ETH    ETH      UCY      UCY      UCY
#        6个不同的大洋

# for desired_source in ['EP', 'NA', 'NI', 'SI', 'SP', 'WP']:
for desired_source in ['WP']:
    print("desired_source:",desired_source)
#for desired_source in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
    for data_class in ['train', 'val', 'test']:
    # for data_class in ['test']:
        print("data_class:", data_class)
        env = Environment(node_type_list=['PEDESTRIAN'], standardization=standardization)
        attention_radius = dict() #存储节点之间的注意半径信息
        attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 3.0
        env.attention_radius = attention_radius #将 attention_radius 应用于环境。

        scenes = []
        #数据字典路径 pkl文件的路径 'processed_data_noise_new/ep_train.pkl'
        data_dict_path = os.path.join(data_folder_name, '_'.join([desired_source, data_class]) + '.pkl') #根据指定的数据路径和文件名，使用 os.path.join 函数构建数据字典路径 data_dict_path。
        #从这里不行了 唉
        print(os.path.join('bst_divi10_train_val_test_inlcude15', desired_source, data_class))

        for subdir, dirs, files in os.walk(os.path.join('bst_divi10_train_val_test_inlcude15', desired_source, data_class)):
            print("subdir:", subdir)
        # for subdir, dirs, files in os.walk(os.path.join('raw_data', desired_source, data_class)):
            #使用 os.walk 函数遍历指定数据源和数据类别的文件夹raw_data/eth/train，
            # os.walk函数接受一个路径作为参数，并生成一个迭代器，迭代器会遍历指定路径下的所有文件夹、子文件夹和文件。
            # 每次迭代，os.walk返回一个三元组，
            # 包含当前文件夹的路径、
            # 当前文件夹中的子文件夹列表
            # 以及当前文件夹中的文件列表
            #subdir变量表示当前文件夹的路径，
            # dirs变量是一个列表，包含当前文件夹中的子文件夹名称，
            # files变量是一个列表，包含当前文件夹中的文件名称
            gph_path = os.path.join('TropiCycloneNet/all_ocean_gph500', desired_source) #/年份/大洋的名字/年yyyy月mm日dd小时hh.npy
            env_path = os.path.join('TropiCycloneNet/all_area_correct_location_includeunder15', desired_source) #/年份/大洋的名字/年yyyy月mm日dd小时hh.npy

            num = 0
            for file in files:
                if file.endswith('.txt'): #通过检查文件后缀为 .txt 的文件来获取数据文件
                    input_data_dict = dict() #创建一个空字典 读取文件内容并进行数据处理
                    full_data_path = os.path.join(subdir, file) #subdir变量表示当前文件夹的路径 files变量是一个列表，包含当前文件夹中的文件名称
                    print('At', full_data_path)
                    num = num+1
                    print("num:",num)
                    data = pd.read_csv(full_data_path, sep='\t', index_col=False, header=None) #使用 pd.read_csv 函数加载文件数据，
                    # 设置分隔符为 \t， 刚好前四列是这个
                    data.columns = ['frame_id', 'track_id', 'pos_x', 'pos_y', 'inten_x', 'inten_y',
                                    'yymmddtt', 'name']#列名为 ['frame_id', 'track_id', 'pos_x', 'pos_y']

                    data['frame_id'] = pd.to_numeric(data['frame_id'], downcast='integer') #将 frame_id 和 track_id 列转换为整数类型
                    data['track_id'] = pd.to_numeric(data['track_id'], downcast='integer')
                    #？？？？？？？？？？？？？？？？？？？？？？？
                    # data['frame_id'] = data['frame_id'] // 10 #将 frame_id 除以 10，并将结果减去最小值，以对时间步长进行调整。可能有问题
                    data['frame_id'] -= data['frame_id'].min() #所以时间步长都是从0开始的
                    data['node_type'] = 'PEDESTRIAN'
                    data['node_id'] = data['track_id'].astype(str) #same in one txt file

                    data.sort_values('frame_id', inplace=True) #根据 frame_id 列对数据进行排序。

                    max_timesteps = data['frame_id'].max()
                    #节点添加到场景中，并根据数据类别进行场景的增强操作。
                    # 如果数据类别为训练数据，将使用不同角度对场景进行增强。
                    # 然后，将场景添加到 scenes 列表中

                    #空的
                    #创建了一个Scene对象，用于表示场景数据。它包含了场景的时间步数、时间步长、名称等信息。
                    scene = Scene(timesteps=max_timesteps+1, dt=dt, name=desired_source + "_" + data_class, aug_func=augment if data_class == 'train' else None)

                    for node_id in pd.unique(data['node_id']):#遍历每个唯一的node_id（节点标识符），
                        # 从数据中选择具有相同node_id的行，并提取出'pos_x'和'pos_y'列的数值。
                        # 如果节点的数据点数少于2，则忽略该节点

                        node_df = data[data['node_id'] == node_id]

                        node_values = node_df[['pos_x', 'pos_y']].values

                        node_values_i = node_df[['inten_x', 'inten_y']].values

                        node_ymdt = node_df[['yymmddtt']].values
                        node_name = node_df[['name']].values

                        if node_values.shape[0] < 2:
                            continue

                        new_first_idx = node_df['frame_id'].iloc[0] #第一行的"frame_id"值

                        x = node_values[:, 0]#横坐标
                        y = node_values[:, 1]#纵坐标
                        vx = derivative_of(x, scene.dt) #dt=0.4
                        vy = derivative_of(y, scene.dt)
                        ax = derivative_of(vx, scene.dt)
                        ay = derivative_of(vy, scene.dt)

                        x_i = node_values_i[:, 0]  # 横坐标
                        y_i = node_values_i[:, 1]  # 纵坐标
                        vx_i = derivative_of(x_i, scene.dt)  # dt=0.4
                        vy_i = derivative_of(y_i, scene.dt)
                        ax_i = derivative_of(vx_i, scene.dt)
                        ay_i = derivative_of(vy_i, scene.dt)

                        gph_data = []
                        env_data = []

                        yy = file[2:6]
                        if(desired_source=='WP' and data_class=='test'):
                            name = file[9:-4]
                        else:
                            name = str(node_name[0])[2:-2]

                        for i in range(node_ymdt.shape[0]):
                            # yy = str(node_ymdt[0])[1:-1][0:4]
                            yymmddtt = str(node_ymdt[i])[1:-1]
                            npy_name = yymmddtt + '.npy'
                            full_gph_path = os.path.join(gph_path, yy, name, npy_name)
                            # print("full_gph_path:", full_gph_path)

                            gph_alone = np.load(full_gph_path)
                            gph_data.append(gph_alone)  # 将读取的数据添加到列表中

                            full_env_path = os.path.join(env_path, yy, name, npy_name)
                            env_alone = np.load(full_env_path,allow_pickle=True).item()
                            env_data.append(env_alone)  # 将读取的数据添加到列表中

                        gph_data = np.array(gph_data)  # 将列表转换为tuple
                        env_data = np.array(env_data)  #env[0]--env[n]

                        # gph1 = 0

                        data_dict = {('position', 'x'): x,
                                     ('position', 'y'): y,
                                     ('velocity', 'x'): vx,
                                     ('velocity', 'y'): vy,
                                     ('acceleration', 'x'): ax,
                                     ('acceleration', 'y'): ay,

                                     ('intensity', 'x'): x_i,
                                     ('intensity', 'y'): y_i,
                                     ('velocity_i', 'x'): vx_i,
                                     ('velocity_i', 'y'): vy_i,
                                     ('acceleration_i', 'x'): ax_i,
                                     ('acceleration_i', 'y'): ay_i

                                     # ,
                                     # ('gph', 'x'): gph,
                                     # ('gph', 'y'): gph1

                                     }
# 要不 把gph单独加到node里面 不在data_dict里了？
                        node_data = pd.DataFrame(data_dict, columns=data_columns)
                        node = Node(node_type=env.NodeType.PEDESTRIAN, node_id=node_id, data=node_data
                                    , gph_data = gph_data
                                    , env_data = env_data
                                    ) #创建一个Node对象来表示该节点，包括节点类型、节点标识符和数据
                        node.first_timestep = new_first_idx #将节点的第一个时间步索引设置为new_first_idx

                        scene.nodes.append(node) #创建的节点添加到场景的节点列表 多个节点 分开了
                    if data_class == 'train': #如果data_class为'train'，则创建一个空列表augmented，然后对场景进行增强操作。
                        # 在这里，使用不同的角度对场景进行增强，每个角度增强一个新的场景，
                        # 并将其添加到augmented列表中。
                        scene.augmented = list()
                        angles = np.arange(0, 360, 15) if data_class == 'train' else [0]
                        for angle in angles:
                            scene.augmented.append(augment_scene(scene, angle))

                    print(scene)
                    scenes.append(scene)
        #这句输出了
        print(f'Processed {len(scenes):.2f} scene for data class {data_class}')

        env.scenes = scenes

        if len(scenes) > 0: # data_dict_path：数据字典路径 pkl文件的路径
            with open(data_dict_path, 'wb') as f:
                dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL) #使用dill.dump函数将env对象（包含所有场景）序列化并写入文件。protocol=dill.HIGHEST_PROTOCOL参数指定了序列化协议，使用dill库的最高协议级别。
                #所有的场景数据就被保存到了文件中，可以在需要时进行加载和使用
exit() #到这就不跑了 所以之前没有处理stanford的数据
#===================================================================================================
# Process Stanford Drone. Data obtained from Y-Net github repo
data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])


for data_class in ["train", "test"]:
    raw_path = "raw_data/stanford" #目的是处理SDD（Stanford Drone Dataset）数据集
    out_path = "processed_data"
    data_path = os.path.join(raw_path, f"{data_class}_trajnet.pkl")
    print(f"Processing SDD {data_class}")
    data_out_path = os.path.join(out_path, f"sdd_{data_class}.pkl")
    df = pickle.load(open(data_path, "rb")) #使用pickle.load函数加载数据集文件，得到数据的DataFrame对象。
    env = Environment(node_type_list=['PEDESTRIAN'], standardization=standardization)
    attention_radius = dict() #创建了一个空字典
    attention_radius[(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN)] = 3.0
    env.attention_radius = attention_radius

    scenes = []

    group = df.groupby("sceneId")

    for scene, data in group:
        data['frame'] = pd.to_numeric(data['frame'], downcast='integer')
        data['trackId'] = pd.to_numeric(data['trackId'], downcast='integer')

        data['frame'] = data['frame'] // 12

        data['frame'] -= data['frame'].min()

        data['node_type'] = 'PEDESTRIAN'
        data['node_id'] = data['trackId'].astype(str)

        # apply data scale as same as PECnet
        data['x'] = data['x']/50
        data['y'] = data['y']/50

        # Mean Position
        data['x'] = data['x'] - data['x'].mean()
        data['y'] = data['y'] - data['y'].mean()

        max_timesteps = data['frame'].max()

        if len(data) > 0:

            scene = Scene(timesteps=max_timesteps+1, dt=dt, name="sdd_" + data_class, aug_func=augment if data_class == 'train' else None)
            n=0
            for node_id in pd.unique(data['node_id']):

                node_df = data[data['node_id'] == node_id]


                if len(node_df) > 1:
                    assert np.all(np.diff(node_df['frame']) == 1)
                    if not np.all(np.diff(node_df['frame']) == 1):
                        pdb.set_trace() #这句可能有点问题 就是说

                    node_values = node_df[['x', 'y']].values

                    if node_values.shape[0] < 2:
                        continue

                    new_first_idx = node_df['frame'].iloc[0]

                    x = node_values[:, 0]
                    y = node_values[:, 1]
                    vx = derivative_of(x, scene.dt)
                    vy = derivative_of(y, scene.dt)
                    ax = derivative_of(vx, scene.dt)
                    ay = derivative_of(vy, scene.dt)

                    data_dict = {('position', 'x'): x,
                                 ('position', 'y'): y,
                                 ('velocity', 'x'): vx,
                                 ('velocity', 'y'): vy,
                                 ('acceleration', 'x'): ax,
                                 ('acceleration', 'y'): ay}

                    node_data = pd.DataFrame(data_dict, columns=data_columns)
                    node = Node(node_type=env.NodeType.PEDESTRIAN, node_id=node_id, data=node_data)
                    node.first_timestep = new_first_idx

                    scene.nodes.append(node)
            if data_class == 'train':
                scene.augmented = list()
                angles = np.arange(0, 360, 15) if data_class == 'train' else [0]
                for angle in angles:
                    scene.augmented.append(augment_scene(scene, angle))

            print(scene)
            scenes.append(scene)
    env.scenes = scenes

    if len(scenes) > 0:
        with open(data_out_path, 'wb') as f:
            #pdb.set_trace()
            dill.dump(env, f, protocol=dill.HIGHEST_PROTOCOL) #场景数据 存入pkl文件 sdd_train.pkl/sdd_test.pkl
