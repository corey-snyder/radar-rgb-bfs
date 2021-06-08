import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
from skimage.io import imread
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
from utils.plot_utils import plot_func
from utils.evaluation_utils import *
import pathlib


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-yaml", help="path of yaml file", type=str)
    args = parser.parse_args()
    yaml_test_path = args.yaml

    # open and use test yaml file
    with open(yaml_test_path) as file:
        setup_dict = yaml.load(file, Loader=yaml.FullLoader)
    runs_dir = setup_dict['run_path']  # path to directory of runs
    test_path = setup_dict['test_path']
    try_gpu = setup_dict['try_gpu']
    step_height = setup_dict['step_height']
    step_width = setup_dict['step_width']
    gt_path = setup_dict['GT']
    pantry_12_flag = setup_dict['pantry_12']
    net_choice = setup_dict['net_choice']

    test_name = pathlib.Path(test_path).parts[-2]
    if pantry_12_flag:
        test_name = 'pantry_12'

    save_dir = runs_dir + '/eval/' + test_name + '_' + net_choice
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    runs = os.listdir(runs_dir)
    runs.remove('eval')
    results = []

    # set up for F-scores
    thresholds = [i * 0.05 for i in range(20)]
    filenames = os.listdir(gt_path)
    gt_images = []
    for ii in range(len(filenames)):
        gt_images.append(imread(os.path.join(gt_path, 'rgb_%i.png' % ii)))

    for run in runs:
        run_path = runs_dir + '/' + run
        print('\n'+run_path)
        if net_choice == 'test':
            net_path = run_path + '/model_bfs_test.pt'
        elif net_choice == 'train':
            net_path = run_path + '/model_bfs_train.pt'
        else:
            raise Exception('Model Option not test or train')

        yaml_train_path = run_path + '/train.yaml'

        with open(yaml_train_path) as file:
            setup_dict = yaml.load(file, Loader=yaml.FullLoader)
        downsample_rate = setup_dict['downsample']
        patch_height = setup_dict['patch_height']
        patch_width = setup_dict['patch_width']
        seq_len = setup_dict['seq_len']

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

        # load data
        D, _, S_target = load_rgb_data(test_path, n_frames=seq_len, rescale_factor=downsample_rate)

        # Swap axes (1 image with many channels)
        D = torch.transpose(D, 0, 1)

        # init model
        model = UNet(n_channels=seq_len, n_classes=seq_len)
        model.load_state_dict(torch.load(net_path))
        model.to(device)
        model.eval()

        output = model(D.to(device))
        output = output.detach().cpu().numpy()

        # Swap axes (1 image with many channels)
        output = np.swapaxes(output, 0, 1)

        if downsample_rate != 1:
            gt_images_run = [gt_images[ii][::int(1 / downsample_rate), ::int(1 / downsample_rate)] for ii in range(len(gt_images))]
        gt_images_run = gt_images_run[:seq_len]

        # extract sparse predictions (channel 1)
        pred_images = [np.array(np.abs(output[ii, 0]) * 255, dtype=np.uint8) for ii in range(seq_len)]  # max abs val of float image is 1

        if pantry_12_flag:
            run_results = compute_metrics(gt_images_run[:12], pred_images[:12], thresholds)
        else:
            run_results = compute_metrics(gt_images_run, pred_images, thresholds)
        results.append(run_results)

    print('\n')
    auc_mean, auc_std = display_dir_results(results, save_dir)
    print(auc_mean, auc_std)
    np.save(save_dir + '/AUC.npy', [auc_mean, auc_std], allow_pickle=True)