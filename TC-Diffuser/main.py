from tc_diffuser import TCDDIFFUSER
import argparse
import os
import yaml
# from pprint import pprint
from easydict import EasyDict
import numpy as np
import pdb
import torch
import random



def parse_args():
    parser = argparse.ArgumentParser(
        description='Pytorch implementation of MID')
    parser.add_argument('--config', default='configs/baseline.yaml')
    parser.add_argument('--train_dataset', default='WP') #['ALL','EP', 'NA', 'NI', 'SI', 'SP', 'WP']
    parser.add_argument('--eval_dataset', default='WP')  # ['ALL','EP', 'NA', 'NI', 'SI', 'SP', 'WP']

    return parser.parse_args()


def main():
    cuda_idx = 0
    device = torch.device('cuda:' + str(cuda_idx))
    torch.cuda.set_device(device)

    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # parse arguments and load config
    args = parse_args()
    with open(args.config,encoding='utf-8') as f:
       config = yaml.safe_load(f)

    for k, v in vars(args).items():
       config[k] = v
    config["exp_name"] = 'baseline_for_WP_WP_1-4-270'

    config["train_dataset"] = args.train_dataset
    config["eval_dataset"] = args.eval_dataset
    #pdb.set_trace()
    config = EasyDict(config)


    if config["eval_mode"]:
        test = [270, 260, 250, 240, 230, 220, 210, 200, 190, 180, 170, 160, 150, 140, 130, 120, 110, 100, 90, 80, 70,
                60, 50, 40, 30, 20, 10]
        # test = [190, 180, 170, 160, 150, 140, 130, 120, 110, 100, 90, 80, 70,
                # 60, 50, 40, 30, 20, 10]
        test = [160]  # 测试时 future都固定值

        for i in test:
            config.eval_at = i
            seed = 123
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)  # Numpy module.
            random.seed(seed)  # Python random module.
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            agent = TCDDIFFUSER(config)

            sampling = "ddim"
            # steps = 5 #原始
            step = 5  # 改 5
            config.eval_at = i
            agent.eval(sampling, 100//step, i)
    else:
        agent = TCDDIFFUSER(config)

        sampling = "ddim"
        # steps = 5
        step = 5
        agent.train()



if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    main()
