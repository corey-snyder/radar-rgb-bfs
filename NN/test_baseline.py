import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import yaml
from datetime import datetime
import shutil
import matplotlib.pyplot as plt
from baseline.model import IstaNet
from utils.data_loader import *
from utils.patch_utils import *
from utils.tensorboard_utils import plot_classes_preds

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-yaml", help="path of yaml file", type=str)
    args = parser.parse_args()
    yaml_test_path = args.yaml

    with open(yaml_test_path) as file:
        setup_dict = yaml.load(file, Loader=yaml.FullLoader)
    run_path = setup_dict['run_path']
    test_path = setup_dict['test_path']
    try_gpu = setup_dict['try_gpu']
    step_height = setup_dict['step_height']
    step_width = setup_dict['step_width']