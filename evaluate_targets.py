import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
import os
from skimage.io import imread
from skimage.transform import rescale
from NN.utils.evaluation_utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-S", help="path of Sparse numpy file", type=str)
    parser.add_argument("-rescale",help="downsample rate per dimension", type=float)
    parser.add_argument("-GT", help="path of dir of GT", type=str)
    args = parser.parse_args()
    s_path = args.S
    rescale_factor = args.rescale
    gt_path = args.GT

    # Load Data
    S = np.load(s_path)

    # Compute F-scores
    thresholds = [i * 0.05 for i in range(20)]
    filenames = os.listdir(gt_path)
    gt_images = []
    for ii in range(len(filenames)):
        gt_images.append(imread(os.path.join(gt_path, 'rgb_%i.png'%ii)))

    if rescale_factor !=1:
        gt_images = [gt_images[ii][::int(1/rescale_factor),::int(1/rescale_factor)] for ii in range(len(gt_images))]
        S = rescale(S,(1,rescale_factor,rescale_factor),anti_aliasing=True)

    # comment below for pantry[:12]
    seq_len = S.shape[0]
    gt_images = gt_images[:seq_len]
    pred_images = [np.array(np.abs(S[ii])*255,dtype=np.uint8) for ii in range(len(S))]  # max abs val of float image is 1

    # # only for pantry[:12]
    # gt_images = gt_images[:12]
    # pred_images = [np.array(np.abs(S[ii]) * 255, dtype=np.uint8) for ii in range(12)]  # max abs val of float image is 1


    print('Computing full metrics...')
    results = compute_metrics(gt_images, pred_images, thresholds)
    display_f_score(results)

    ROC_curve(results,True)


