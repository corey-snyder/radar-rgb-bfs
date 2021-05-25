import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


def plot_classes_preds(output,L, S):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    L_pred = output[:,0]
    S_pred = output[:,1]

    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 25))
    # plt.suptitle("Original Results")
    for idx,frame in enumerate(np.arange(0,30,3)):
        ax = fig.add_subplot(10, 4, idx*4 + 1, xticks=[], yticks=[])
        plt.imshow(L[frame, 0],cmap='gray')
        if idx==0: plt.title('L Target')
        ax = fig.add_subplot(10, 4, idx*4 + 2, xticks=[], yticks=[])
        plt.imshow(L_pred[frame],cmap='gray')
        if idx==0: plt.title('L Prediction')
        ax = fig.add_subplot(10, 4, idx*4 + 3, xticks=[], yticks=[])
        plt.imshow(S[frame,0],cmap='gray', vmin=-1, vmax=1)
        if idx==0: plt.title('S Target')
        ax = fig.add_subplot(10, 4, idx*4 + 4, xticks=[], yticks=[])
        plt.imshow(S_pred[frame],cmap='gray', vmin=-1, vmax=1)
        if idx==0: plt.title('S Prediction')
    plt.tight_layout()
    return fig


def plot_classes_preds_unet(output, D, S):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''

    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 25))
    # plt.suptitle("Original Results")
    for idx,frame in enumerate(np.arange(0,30,3)):
        ax = fig.add_subplot(10, 3, idx*3 + 1, xticks=[], yticks=[])
        plt.imshow(D[0, frame],cmap='gray')
        if idx==0: plt.title('Input')
        ax = fig.add_subplot(10, 3, idx*3 + 2, xticks=[], yticks=[])
        plt.imshow(S[0,frame],cmap='gray')
        if idx==0: plt.title('S Target')
        ax = fig.add_subplot(10, 3, idx*3 + 3, xticks=[], yticks=[])
        plt.imshow(np.abs(output[0, frame]),cmap='gray', vmin=0, vmax=1)
        if idx==0: plt.title('S Prediction')
    plt.tight_layout()
    return fig
