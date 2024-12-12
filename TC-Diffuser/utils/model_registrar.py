import os
import torch
import torch.nn as nn


def get_model_device(model):
    return next(model.parameters()).device

#用于管理不同类型的模型并在需要时动态地获取或添加这些模型。
class ModelRegistrar(nn.Module):
    #接受model_dir（模型存储路径） 加载模型？
    def __init__(self, model_dir, device):
        super(ModelRegistrar, self).__init__()
        self.model_dict = nn.ModuleDict() #空的 用于存储多个模型并以字典形式进行管理 将用于存储多个子模型
        self.model_dir = model_dir
        self.device = device

    def forward(self):
        #因为ModelRegistrar的主要目的不是进行前向传播，而是存储模型参数。
        raise NotImplementedError('Although ModelRegistrar is a nn.Module, it is only to store parameters.')

    # 这是获取模型的方法。
    # 它接受一个模型名称name和一个可选的model_if_absent参数。
    # 它根据name来查找已有的模型并返回，如果不存在，则根据model_if_absent创建新的模型并返回
    def get_model(self, name, model_if_absent=None):
        # 4 cases: name in self.model_dict and model_if_absent is None         (OK)
        #          name in self.model_dict and model_if_absent is not None     (OK)
        #          name not in self.model_dict and model_if_absent is not None (OK)
        #          name not in self.model_dict and model_if_absent is None     (NOT OK)

        if name in self.model_dict:
            return self.model_dict[name]

        elif model_if_absent is not None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model_dict[name] = model_if_absent.to(self.device)
            return self.model_dict[name]

        else:
            raise ValueError(f'{name} was never initialized in this Registrar!')

    # 这是根据模型名称的匹配查找模型的方法。它接受一个模型名称name，并返回名称中包含name的所有模型的列表。
    def get_name_match(self, name):
        ret_model_list = nn.ModuleList()
        for key in self.model_dict.keys():
            if name in key:
                ret_model_list.append(self.model_dict[key])
        return ret_model_list

    # 这是根据模型名称的不匹配查找模型的方法。它接受一个模型名称name，并返回名称中不包含name的所有模型的列表。
    def get_all_but_name_match(self, name):
        ret_model_list = nn.ModuleList()
        for key in self.model_dict.keys():
            if name not in key:
                ret_model_list.append(self.model_dict[key])
        return ret_model_list

    # 打印当前ModelRegistrar实例中存储的所有模型名称的方法。
    def print_model_names(self):
        print(self.model_dict.keys())

    # 这是将当前存储在ModelRegistrar实例中的所有模型保存到指定路径的方法。
    def save_models(self, save_path):
        # Create the model directiory if it's not present.
        # save_path = os.path.join(self.model_dir,
        #                          'model_registrar-%d.pt' % curr_iter)

        torch.save(self.model_dict, save_path)

    # 从给定的模型字典model_dict中加载所有模型到ModelRegistrar实例的方法。
    def load_models(self, model_dict):
        self.model_dict.clear()

        #save_path = os.path.join(self.model_dir,
        #                         'model_registrar-%d.pt' % iter_num)

#        print('')
        print('Loading Encoder')
        self.model_dict = model_dict


    def to(self, device):
        for name, model in self.model_dict.items():
            if get_model_device(model) != device:
                model.to(device)
