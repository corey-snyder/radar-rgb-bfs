import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
import os
from skimage.io import imread
from evaluate import compute_metrics, display_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-S", help="path of Sparse numpy file", type=str)
    parser.add_argument("-ds",help="downsample rate per dimension", type=float)
    parser.add_argument("-GT", help="path of dir of GT", type=str)
    args = parser.parse_args()
    s_path = args.S
    downsample_rate = args.ds
    gt_path = args.GT

    # Load Data
    S = np.load(s_path)   
    # Compute F-scores

    thresholds = [i * 0.05 for i in range(20)]
    filenames = os.listdir(gt_path)
    gt_images = []
    for ii in range(len(filenames)):
        gt_images.append(imread(os.path.join(gt_path, 'rgb_%i.png'%ii)))
    if downsample_rate !=1:
        gt_images = [gt_images[ii][::int(1/downsample_rate),::int(1/downsample_rate)] for ii in range(len(gt_images))]
        S = S[:,::int(1/downsample_rate),::int(1/downsample_rate)]
    output_len = 30 #S.shape[0]
    gt_images = gt_images[:output_len]
    pred_images = [np.array(np.abs(S[ii])*255,dtype=np.uint8) for ii in range(output_len)]  # max abs val of float image is 1
    
    print('Computing full metrics...')
    results = compute_metrics(gt_images, pred_images, thresholds, False)
    display_results(results)

    print('\n\nComputing subsampled metrics...')
    results = compute_metrics(gt_images, pred_images, thresholds, True)
    display_results(results)
