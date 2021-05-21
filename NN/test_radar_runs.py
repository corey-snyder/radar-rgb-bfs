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
from radar.model import IstaNet
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

    test_name = pathlib.Path(test_path).parts[-2]
    if pantry_12_flag:
        test_name = 'pantry_12'

    runs = os.listdir(runs_dir)
    try: runs.remove('eval')
    except: pass
    runs_in = [run for run in runs if 'RADAR_in' in run]
    runs_before = [run for run in runs if 'RADAR_before' in run]
    runs_after = [run for run in runs if 'RADAR_after' in run]

    # set up for F-scores
    thresholds = [i * 0.05 for i in range(20)]
    filenames = os.listdir(gt_path)
    gt_images = []
    for ii in range(len(filenames)):
        gt_images.append(imread(os.path.join(gt_path, 'rgb_%i.png' % ii)))

    for run_category_list, run_category_name in zip([runs_in, runs_before, runs_after],['in/','before/','after/']):

        save_dir = runs_dir + '/eval/' + run_category_name + test_name
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        results = []
        for run in run_category_list:
            run_path = runs_dir + '/' + run
            print('\n'+run_path)
            net_path = run_path + '/model_bfs.pt'
            yaml_train_path = run_path + '/train.yaml'

            with open(yaml_train_path) as file:
                setup_dict = yaml.load(file, Loader=yaml.FullLoader)
            n_layers = setup_dict['n_layers']
            downsample_rate = setup_dict['downsample']
            patch_height = setup_dict['patch_height']
            patch_width = setup_dict['patch_width']
            seq_len = setup_dict['seq_len']
            radar_inclusion_type = setup_dict['radar_inclusion_type']

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
            D, L_target, S_target, R = load_radar_rgb_data(test_path, n_frames=seq_len, rescale_factor=downsample_rate)
            data_shape = list(np.concatenate([D.shape[:2], [patch_height, patch_width]]))
            step_shape = (step_height, step_width)

            model = IstaNet(data_shape, n_layers, radar_inclusion_type)
            model.load_state_dict(torch.load(net_path))
            model.to(device)
            model.eval()

            output = infer_full_image(D, model, data_shape, step_shape, device, R=R)
            output, D = output.detach().cpu().numpy(), D.detach().cpu().numpy()

            if downsample_rate != 1:
                gt_images_run = [gt_images[ii][::int(1 / downsample_rate), ::int(1 / downsample_rate)] for ii in range(len(gt_images))]
            gt_images_run = gt_images_run[:seq_len]

            # extract sparse predictions (channel 1)
            pred_images = [np.array(np.abs(output[ii, 1]) * 255, dtype=np.uint8) for ii in range(seq_len)]  # max abs val of float image is 1
            if pantry_12_flag:
                run_results = compute_metrics(gt_images_run[:12], pred_images[:12], thresholds)
            else:
                run_results = compute_metrics(gt_images_run, pred_images, thresholds)
            results.append(run_results)

        print('\n')
        auc_mean, auc_std = display_dir_results(results, save_dir)
        print(auc_mean, auc_std)
        np.save(save_dir + '/AUC.npy', [auc_mean, auc_std], allow_pickle=True)


