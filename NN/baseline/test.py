import numpy as np
import matplotlib.pyplot as plt
import torch
from model import IstaNet
from train import load_data
import argparse
import yaml
import os
from skimage.io import imread
from skimage.transform import rescale
from evaluate import compute_metrics


def plot_func(output,D):
    n = 3
    N = len(D)//n
    plt.figure(figsize=(5,35))
    for ii in range(N):
        plt.subplot(N,3,3*ii+1)
        plt.imshow(D[n*ii,0],cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if ii == 0: plt.title('Original')

        plt.subplot(N, 3, 3 * ii + 2)
        plt.imshow(output[n*ii, 0],cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if ii == 0: plt.title('L')

        plt.subplot(N, 3, 3 * ii + 3)
        plt.imshow(np.abs(output[n*ii, 1]),cmap='gray')
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
    test_path = setup_dict['test_path'][0]
    try_gpu = setup_dict['try_gpu'][0]
    if 'GT' in setup_dict:
        print('Testing F-score')
        fscore_flag = True
        gt_path = setup_dict['GT'][0]

    net_path = run_path + '/model_bfs.pt'
    yaml_train_path = run_path + '/setup.yaml'

    with open(yaml_train_path) as file:
        setup_dict = yaml.load(file, Loader=yaml.FullLoader)
    n_layers = setup_dict['n_layers'][0]
    downsample_rate = setup_dict['downsample'][0]

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

    D, L_target, S_target = load_data(test_path,rescale_factor=downsample_rate)
    L_test = torch.zeros_like(D)
    S_test = torch.zeros_like(D)
    # move tensors to GPU if CUDA is available
    D, L_test, S_test = D.to(device), L_test.to(device), S_test.to(device)

    (n_frames,n_channels,im_height,im_width) = D.shape
    model = IstaNet(D.shape,n_layers)
    model.load_state_dict(torch.load(net_path))
    model.to(device)
    model.eval()

    output = model(D,L_test,S_test)
    output, D = output.detach().cpu().numpy(), D.detach().cpu().numpy()

    plot_func(output,D)
    plt.show()

    # Compute F-scores

    # thresholds = [i * 0.05 for i in range(20)]
    # filenames = os.listdir(gt_path)
    # gt_images = []
    # for ii in range(len(filenames)):
    #     gt_images.append(imread(os.path.join(gt_path, 'rgb_%i.png'%ii)))
    # if downsample_rate !=1:
    #     gt_images = [gt_images[ii][::int(1/downsample_rate),::int(1/downsample_rate)] for ii in range(len(gt_images))]
    # output_len = output.shape[0]
    # gt_images = gt_images[:output_len]
    # pred_images = [np.array(np.abs(output[ii,1])*255,dtype=np.uint8) for ii in range(output_len)]  # max abs val of float image is 1
    #
    # compute_metrics(gt_images, pred_images, thresholds, False)
    # pass