import os
import numpy as np

from argparse import ArgumentParser
from skimage.io import imread

def load_images(gt_path, pred_path):
    filenames = os.listdir(gt_path)
    gt_images = []
    pred_images = []
    for f in filenames:
        gt_images.append(imread(os.path.join(gt_path, f)))
        pred_images.append(imread(os.path.join(pred_path, f)))
    return gt_images, pred_images

def prepare_gt(image, subsample):
    if not subsample:
        return image > 0
    else:
        max_value = np.max(image)
        return image == max_value

def display_results(results_dict):
    for threshold in np.sort(list(results_dict.keys())):
        print('Metrics @ threshold {:.2f}:'.format(threshold))
        print('Precision: {:.3f} | Recall: {:.3f} | F-measure: {:.3f}'.format(results_dict[threshold]['precision'],
                                                                              results_dict[threshold]['recall'],
                                                                              results_dict[threshold]['f-measure']))

def compute_metrics(gt_images, pred_images, thresholds, subsample):
    count_dict = {}
    for t in thresholds:
        count_dict[t] = {'true_positive': 0, 'predicted_positive': 0, 'gt_positive': 0}
    for i in range(len(gt_images)):
        gt_image = gt_images[i]
        gt_image = prepare_gt(gt_image, subsample)
        for t in thresholds:
            pred_image = pred_images[i]
            pred_image = (pred_image/255) > t # assume images are saved as np.uint8 type
            count_dict[t]['true_positive'] += np.sum(np.logical_and(pred_image, gt_image))
            count_dict[t]['predicted_positive'] += np.sum(pred_image)
            count_dict[t]['gt_positive'] += np.sum(gt_image)
    results_dict = {}
    for t in thresholds:
        results_dict[t] = {}
        results_dict[t]['precision'] = count_dict[t]['true_positive']/count_dict[t]['predicted_positive']
        results_dict[t]['recall'] = count_dict[t]['true_positive']/count_dict[t]['gt_positive']
        results_dict[t]['f-measure'] = 2*results_dict[t]['precision']*results_dict[t]['recall']/(results_dict[t]['precision']+results_dict[t]['recall']+1e-9)
    return results_dict

if __name__ == '__main__':
    '''
    Example usage:
    
    python3 evaluate.py --gt_path RGB_images/groundtruth --pred_path RGB_images/rpca_preds
    '''
    
    # parse input argument(s)
    parser = ArgumentParser()
    parser.add_argument('--gt_path', type=str, help='path to folder with ground-truth images', required=True)
    parser.add_argument('--pred_path', type=str, help='path to folder with prediction images', required=True)
    args = parser.parse_args()
    
    # prepare input arguments
    gt_path = args.gt_path
    pred_path = args.pred_path
    gt_images, pred_images = load_images(gt_path, pred_path)
    thresholds = [i*0.05 for i in range(20)]

    # compute metrics
    print('Computing full metrics...')
    full_results = compute_metrics(gt_images, pred_images, thresholds, False)
    display_results(full_results)
    print('')
    print('Computing subsampled metrics...')
    subsampled_results = compute_metrics(gt_images, pred_images, thresholds, True)
    display_results(subsampled_results)

