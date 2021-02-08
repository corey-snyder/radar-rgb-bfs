import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from model import IstaNet
from train import load_data
import argparse
import yaml


def plot_func(output,D):
    n = 3
    N = len(D)//n
    plt.figure()
    plt.tight_layout()
    for ii in range(N):
        plt.subplot(N,3,3*ii+1)
        plt.imshow(D[n*ii,0])
        plt.xticks([])
        plt.yticks([])
        if ii == 0:
            plt.title('Original')

        plt.subplot(N, 3, 3 * ii + 2)
        plt.imshow(output[n*ii, 0])
        plt.xticks([])
        plt.yticks([])
        if ii == 0:
            plt.title('L')

        plt.subplot(N, 3, 3 * ii + 3)
        plt.imshow(np.abs(output[n*ii, 1]))
        plt.xticks([])
        plt.yticks([])
        if ii == 0:
            plt.title('abs(S)')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-yaml", help="path of yaml file", type=str)
    args = parser.parse_args()
    yampl_path = args.yaml

    with open(yampl_path) as file:
        setup_dict = yaml.load(file, Loader=yaml.FullLoader)
    net_path = setup_dict['net_path'][0]
    test_path = setup_dict['test_path'][0]
    n_layers = setup_dict['n_layers'][0]
    try_gpu = setup_dict['try_gpu'][0]

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

    D, L_target, S_target = load_data(test_path,rescale_factor=.125)
    L_test = torch.zeros_like(D)
    S_test = torch.zeros_like(D)


    (n_frames,n_channels,im_height,im_width) = D.shape
    model = IstaNet(D.shape,12)
    model.load_state_dict(torch.load(net_path))
    model.eval()

    output = model(D,L_test,S_test)
    output = output.detach().numpy()
    D = D.detach().numpy()

    plot_func(output,D)
    plt.show()
