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
from baseline.model import IstaNet
from utils.data_loader import *
from utils.patch_utils import *
from utils.plot_utils import plot_func
from utils.evaluation_utils import *
from utils.tensorboard_utils import plot_classes_preds

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
    step_height = setup_dict['step_height']
    step_width = setup_dict['step_width']

    # if groundtruth is linked, provide F-score and AUC/ROC results
    if 'GT' in setup_dict:
        print('Computing Quantitative Results')
        quant_flag = True
        gt_path = setup_dict['GT']
    else: quant_flag = False

    net_path = run_path + '/model_bfs.pt'
    yaml_train_path = run_path + '/train.yaml'

    # open and train yaml file to get details about network and input details
    with open(yaml_train_path) as file:
        setup_dict = yaml.load(file, Loader=yaml.FullLoader)
    n_layers = setup_dict['n_layers']
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
    D, L_target, S_target = load_rgb_data(test_path, n_frames=seq_len, rescale_factor=downsample_rate)
    data_shape = list(np.concatenate([D.shape[:2], [patch_height, patch_width]]))
    step_shape = (step_height, step_width)

    # (n_frames, n_channels, im_height, im_width) = D.shape
    model = IstaNet(data_shape, n_layers)
    model.load_state_dict(torch.load(net_path))
    model.to(device)
    model.eval()

    output = infer_full_image(D, model, data_shape, step_shape, device)
    output, D = output.detach().cpu().numpy(), D.detach().cpu().numpy()

    plot_func(output, D)

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

        # extract sparse predictions (channel 1)
        pred_images = [np.array(np.abs(output[ii, 1]) * 255, dtype=np.uint8) for ii in range(seq_len)]  # max abs val of float image is 1

        results = compute_metrics(gt_images, pred_images, thresholds)
        display_f_score(results)

        # Plot ROC Curve and Compute AUC
        ROC_curve(results, True)

    plt.show()