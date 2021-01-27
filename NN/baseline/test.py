import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from model import IstaNet
from train import load_data


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

    D, L_target, S_target = load_data('/home/spencer/research/radar-rgb-bfs/output/eceb_lobby')
    L_test = torch.zeros_like(D)
    S_test = torch.zeros_like(D)


    (n_frames,n_channels,im_height,im_width) = D.shape
    model = IstaNet(im_height,im_width)
    model.load_state_dict(torch.load('/home/spencer/research/radar-rgb-bfs/NN/baseline/model_bfs.pt'))
    model.eval()

    output = model(D,L_test,S_test)
    output = output.detach().numpy()
    D = D.detach().numpy()

    plot_func(output,D)
    plt.show()