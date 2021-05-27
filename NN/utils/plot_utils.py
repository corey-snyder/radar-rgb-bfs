import matplotlib.pyplot as plt
import numpy as np


def plot_func(output, D):
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
        plt.imshow(np.abs(output[n*ii, 1]),cmap='gray',vmin=0,vmax=1)
        plt.xticks([])
        plt.yticks([])
        if ii == 0: plt.title('abs(S)')
    plt.tight_layout()


def plot_func_radar(output, D, R):
    n = 3
    N = len(D)//n
    plt.figure(figsize=(8,35))
    for ii in range(N):
        plt.subplot(N, 4, 4*ii+1)
        plt.imshow(D[n*ii,0],cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if ii == 0: plt.title('Original')

        plt.subplot(N, 4, 4 * ii + 2)
        plt.imshow(output[n*ii, 0],cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if ii == 0: plt.title('L')

        plt.subplot(N, 4, 4 * ii + 3)
        plt.imshow(np.abs(output[n*ii, 1]),cmap='gray',vmin=0,vmax=1)
        plt.xticks([])
        plt.yticks([])
        if ii == 0: plt.title('abs(S)')

        plt.subplot(N, 4, 4 * ii + 4)
        plt.plot(R[n * ii, 0])
        plt.xticks([])
        plt.yticks([])
        plt.ylim([0,np.max(R.numpy())])
        if ii == 0: plt.title('abs(S)')

    plt.tight_layout()


def plot_func_unet(output, S, D):
    n = 3
    N = len(D) // n
    plt.figure(figsize=(5, 35))
    for ii in range(N):
        plt.subplot(N, 3, 3 * ii + 1)
        plt.imshow(D[n * ii, 0], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if ii == 0: plt.title('Original')

        plt.subplot(N, 3, 3 * ii + 2)
        plt.imshow(abs(S[n * ii, 0]), cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if ii == 0: plt.title('S Target')

        plt.subplot(N, 3, 3 * ii + 3)
        plt.imshow(np.abs(output[n * ii, 0]), cmap='gray', vmin=0, vmax=1)
        plt.xticks([])
        plt.yticks([])
        if ii == 0: plt.title('abs(S)')
    plt.tight_layout()