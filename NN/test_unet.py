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
from unet.unet_model import UNet
from utils.data_loader import *
from utils.patch_utils import *
from utils.tensorboard_utils import *
from utils.plot_utils import plot_func_unet
import os
from utils.evaluation_utils import *



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-yaml", help="path of yaml file", type=str)
    args = parser.parse_args()
    yaml_test_path = args.yaml

    # open and use test yaml file
    with open(yaml_test_path) as file:
        setup_dict = yaml.load(file, Loader=yaml.FullLoader)
    run_path = setup_dict['run_path']
    test_path = setup_dict['test_path']
    try_gpu = setup_dict['try_gpu']
    net_choice = setup_dict['net_choice']

    # if groundtruth is linked, provide F-score and AUC/ROC results
    if 'GT' in setup_dict:
        print('Computing Quantitative Results')
        quant_flag = True
        gt_path = setup_dict['GT']
    else: quant_flag = False

    if net_choice == 'test':
        net_path = run_path + '/model_bfs_test.pt'
    elif net_choice == 'train':
        net_path = run_path + '/model_bfs_train.pt'
    else:
        raise Exception('Model Option not test or train')

    yaml_train_path = run_path + '/train.yaml'

    # open and train yaml file to get details about network and input details
    with open(yaml_train_path) as file:
        setup_dict = yaml.load(file, Loader=yaml.FullLoader)
    downsample_rate = setup_dict['downsample']
    seq_len = setup_dict['seq_len']
    thresh = setup_dict['thresh']

    # check if CUDA is available
    if try_gpu:
        train_on_gpu = torch.cuda.is_available()
    else:
        train_on_gpu = False

    device = torch.device("cuda" if train_on_gpu else "cpu")
    if not train_on_gpu:
        print('CUDA is not available.  Testing on CPU ...')
    else:
        print('CUDA is available!  Testing on GPU ...')

    # load Data (not dataset object since no train/test split)
    D, _, S_target = load_rgb_data(test_path, seq_len, rescale_factor=downsample_rate)

    # Threshold Data
    S_target = threshold_data(S_target, thresh)

    # Swap axes (1 image with many channels)
    D = torch.transpose(D, 0, 1)
    S_target = torch.transpose(S_target, 0, 1)

    # init model
    model = UNet(n_channels=seq_len, n_classes=seq_len)
    model.load_state_dict(torch.load(net_path))
    model.to(device)
    model.eval()

    output = model(D.to(device))
    output = output.detach().cpu().numpy()

    # Swap axes (1 image with many channels)
    D = np.swapaxes(D, 0, 1)
    S_target = np.swapaxes(S_target, 0, 1)
    output = np.swapaxes(output, 0, 1)

    plot_func_unet(output, S_target, D)
    # Compute Quantitative Results
    if quant_flag:

        # F-score
        thresholds = [i * 0.05 for i in range(20)]
        filenames = os.listdir(gt_path)
        gt_images = []
        for ii in range(len(filenames)):
            gt_images.append(imread(os.path.join(gt_path, 'rgb_%i.png' % ii)))
        if downsample_rate != 1:
            gt_images = [gt_images[ii][::int(1 / downsample_rate), ::int(1 / downsample_rate)] for ii in range(len(gt_images))]
        gt_images = gt_images[:seq_len]

        # extract sparse predictions
        pred_images = [np.array(np.abs(output[ii, 0]) * 255, dtype=np.uint8) for ii in range(seq_len)]  # max abs val of float image is 1

        results = compute_metrics(gt_images, pred_images, thresholds)
        display_f_score(results)

        # Plot ROC Curve and Compute AUC
        ROC_curve(results, True)

    plt.show()