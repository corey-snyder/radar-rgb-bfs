import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

def plot_classes_preds(output,L, S, R):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    L_pred = output[:,0]
    S_pred = output[:,1]
    R_min = np.min(R)
    R_max = np.max(R)

    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(18, 25))
    # plt.suptitle("Radar Results")
    for idx,frame in enumerate(np.arange(0,30,3)):
        ax = fig.add_subplot(10, 5, idx*5 + 1, xticks=[], yticks=[])
        plt.imshow(L[frame, 0],cmap='gray')
        if idx==0: plt.title('L Target')
        ax = fig.add_subplot(10, 5, idx*5 + 2, xticks=[], yticks=[])
        plt.imshow(L_pred[frame],cmap='gray')
        if idx==0: plt.title('L Prediction')
        ax = fig.add_subplot(10, 5, idx*5 + 3, xticks=[], yticks=[])
        plt.imshow(S[frame,0],cmap='gray')
        if idx==0: plt.title('S Target')
        ax = fig.add_subplot(10, 5, idx*5 + 4, xticks=[], yticks=[])
        plt.imshow(S_pred[frame],cmap='gray')
        if idx==0: plt.title('S Prediction')
        ax = fig.add_subplot(10, 5, idx*5 + 5, xticks=[])
        ax.yaxis.tick_right()
        plt.plot(R[frame,0])
        plt.ylim([R_min,R_max])
        if idx==0: plt.title('Radar Sparse Likelihood')
    plt.tight_layout()
    return fig