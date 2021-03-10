import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

def plot_classes_preds(output,D, S):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    S_pred = output[:,0]

    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(6, 23))
    # plt.suptitle("Original Results")
    for idx,frame in enumerate(np.arange(0,30,3)):
        ax = fig.add_subplot(10, 3, idx*3 + 1, xticks=[], yticks=[])
        plt.imshow(D[frame, 0],cmap='gray')
        if idx==0: plt.title('Input')
        ax = fig.add_subplot(10, 3, idx*3 + 2, xticks=[], yticks=[])
        plt.imshow(S[frame,0],cmap='gray', vmin=0, vmax=1)
        if idx==0: plt.title('S Target')
        ax = fig.add_subplot(10, 3, idx*3 + 3, xticks=[], yticks=[])
        plt.imshow(S_pred[frame],cmap='gray', vmin=0, vmax=1)
        if idx==0: plt.title('S Prediction')
    plt.tight_layout()
    return fig
