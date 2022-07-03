import numpy as np
import matplotlib.pyplot as plt
import torch
from unet_model import UNet
from train import load_data
import argparse
import yaml
from train import infer_full_image
import os
from skimage.io import imread
import sys
from evaluate import compute_metrics, display_results


def plot_func(output,D, target):
    n = 3
    N = D.shape[1]//n
    plt.figure(figsize=(5,35))
    for ii in range(N):
        plt.subplot(N,3,3*ii+1)
        plt.imshow(D[0, n*ii],cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if ii == 0: plt.title('Original')

        plt.subplot(N, 3, 3 * ii + 2)
        plt.imshow(np.abs(target[0, n*ii]),cmap='gray',vmin=0,vmax=1)
        plt.xticks([])
        plt.yticks([])
        if ii == 0: plt.title('Target')

        plt.subplot(N, 3, 3 * ii + 3)
        plt.imshow(np.abs(output[0, n*ii]),cmap='gray',vmin=0,vmax=1)
        plt.xticks([])
        plt.yticks([])
        if ii == 0: plt.title('abs(S)')

    plt.tight_layout()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-yaml", help="path of yaml file", type=str)
    args = parser.parse_args()
    yampl_test_path = args.yaml

    with open(yampl_test_path) as file:
        setup_dict = yaml.load(file, Loader=yaml.FullLoader)
    run_path = setup_dict['run_path'][0]
    net_name = setup_dict['net_name'][0]
    test_path = setup_dict['test_path'][0]
    try_gpu = setup_dict['try_gpu'][0]
    step_height = setup_dict['step_height'][0]
    step_width = setup_dict['step_width'][0]

    if 'GT' in setup_dict:
        print('Testing F-score')
        fscore_flag = True
        gt_path = setup_dict['GT'][0]
    else: fscore_flag=False

    net_path = run_path + '/' + net_name
    yaml_train_path = run_path + '/setup.yaml'

    with open(yaml_train_path) as file:
        setup_dict = yaml.load(file, Loader=yaml.FullLoader)
    # n_layers = setup_dict['n_layers'][0]
    downsample_rate = setup_dict['downsample'][0]
    patch_height = setup_dict['patch_height'][0]
    patch_width = setup_dict['patch_width'][0]
    threshold = setup_dict['threshold'][0]

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

    D, S_target = load_data(test_path, threshold=threshold, rescale_factor=downsample_rate)
    data_shape = list(np.concatenate([D.shape[:2],[patch_height,patch_width]]))
    step_shape = (step_height, step_width)

    (n_frames,n_channels,im_height,im_width) = D.shape
    model = UNet(n_channels=30, n_classes=30)
    model.load_state_dict(torch.load(net_path))
    model.to(device)
    model.eval()

    output = infer_full_image(D, model, data_shape, step_shape, device)
    output, D = output.detach().cpu().numpy(), D.detach().cpu().numpy()

    plot_func(output,D, S_target)
    plt.show()


    if fscore_flag == False: sys.exit()
    # Compute F-scores

    thresholds = [i * 0.05 for i in range(20)]
    filenames = os.listdir(gt_path)
    gt_images = []
    for ii in range(len(filenames)):
        gt_images.append(imread(os.path.join(gt_path, 'rgb_%i.png'%ii)))
    if downsample_rate !=1:
        gt_images = [gt_images[ii][::int(1/downsample_rate),::int(1/downsample_rate)] for ii in range(len(gt_images))]
    output_len = output.shape[0]
    gt_images = gt_images[:output_len]
    pred_images = [np.array(np.abs(output[ii,0])*255,dtype=np.uint8) for ii in range(output_len)]  # max abs val of float image is 1

    print('Computing full metrics...')
    results = compute_metrics(gt_images, pred_images, thresholds, False)
    display_results(results)

    print('\n\nComputing subsampled metrics...')
    results = compute_metrics(gt_images, pred_images, thresholds, True)
    display_results(results)
