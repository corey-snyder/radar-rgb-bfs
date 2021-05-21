import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate, integrate
from argparse import ArgumentParser
from skimage.io import imread
from sklearn import metrics
import sys


def load_images(gt_path, pred_path):
    filenames = os.listdir(gt_path)
    gt_images = []
    pred_images = []
    for f in filenames:
        gt_images.append(imread(os.path.join(gt_path, f)))
        pred_images.append(imread(os.path.join(pred_path, f)))
    return gt_images, pred_images


def prepare_gt(image):
    return image > 0


def display_f_score(results_dict):
    for threshold in np.sort(list(results_dict.keys())):
        print('Metrics @ threshold {:.2f}:'.format(threshold))
        print('Precision: {:.3f} | Recall: {:.3f} | F-measure: {:.3f}'.format(results_dict[threshold]['precision'],
                                                                              results_dict[threshold]['recall'],
                                                                              results_dict[threshold]['f-measure']))


def compute_metrics(gt_images, pred_images, thresholds):
    # gt_images = gt_images[:12]
    # pred_images = pred_images[:12]
    count_dict = {}
    for t in thresholds:
        count_dict[t] = {'true_positive': 0, 'predicted_positive': 0, 'gt_positive': 0, 'false_positive': 0,
                         'true_negative': 0, 'false_negative': 0}
    for i in range(len(gt_images)):
        gt_image = gt_images[i]
        gt_image = prepare_gt(gt_image)
        gt_images[i] = gt_image

        for t in thresholds:
            pred_image = pred_images[i]
            pred_image = (pred_image / 255) > t  # assume images are saved as np.uint8 type
            count_dict[t]['true_positive'] += np.sum(np.logical_and(pred_image, gt_image))
            count_dict[t]['predicted_positive'] += np.sum(pred_image)
            count_dict[t]['gt_positive'] += np.sum(gt_image)

    fpr, tpr, _ = metrics.roc_curve(np.array(gt_images).flatten(), np.array(pred_images).flatten())
    auc = metrics.roc_auc_score(np.array(gt_images).flatten(), np.array(pred_images).flatten())
    results_dict = {}
    for t in thresholds:
        results_dict[t] = {}
        results_dict[t]['precision'] = count_dict[t]['true_positive'] / count_dict[t]['predicted_positive']
        results_dict[t]['recall'] = count_dict[t]['true_positive'] / count_dict[t]['gt_positive']
        results_dict[t]['f-measure'] = 2 * results_dict[t]['precision'] * results_dict[t]['recall'] / (
                    results_dict[t]['precision'] + results_dict[t]['recall'] + 1e-9)
        results_dict[t]['TPR'] = tpr
        results_dict[t]['FPR'] = fpr
        results_dict[t]['AUC'] = auc
    return results_dict


def ROC_curve(results_dict, plot=False):
    tpr = results_dict[0]['TPR']
    fpr = results_dict[0]['FPR']
    auc = results_dict[0]['AUC']

    if plot:
        plt.figure()
        plt.plot(fpr, tpr)
        plt.title('ROC Curve, AUC: {:.3f}'.format(auc))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()

    return fpr, tpr, auc


def display_dir_results(results_list, save_path):
    original_stdout = sys.stdout
    with open(save_path + '/f-score.txt', 'w') as f:
        sys.stdout = f

        keys = results_list[0].keys()
        all_FPRS = []
        all_TPRS = []
        all_AUCS = []
        plt.figure()
        for key in keys:
            vals = []
            for run in results_list:
                vals.append(run[key]['f-measure'])
                fpr, tpr, auc = ROC_curve(run, False)
                plt.plot(fpr, tpr, '--', alpha=1)
                all_FPRS.append(fpr)
                all_TPRS.append(tpr)
                all_AUCS.append(auc)

            if True in np.isnan(np.array(vals)): print('F-score @ threshold {:.2f}: CONTAINS NAN(s)'.format(key))
            else: print('F-score @ threshold {:.2f}:'.format(key))
            print('Min: {:.3f} | Max: {:.3f} | Mean: {:.3f} +/- {:.3f}'.format(np.nanmin(vals),np.nanmax(vals),
                                                                               np.nanmean(vals), np.nanstd(vals)))
    sys.stdout = original_stdout  # Reset the standard output to its original value
    auc_mean = np.mean(all_AUCS)
    auc_std = np.std(all_AUCS)

    plt.title('ROC Curves AUC = m:{:.3f},std:{:.3f}'.format(np.mean(all_AUCS),np.std(all_AUCS)))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.show()

    if save_path:
        plt.savefig(save_path + '/ROC.png')

    return auc_mean, auc_std


if __name__ == '__main__':
    '''
    Example usage:

    python3 evaluation_utils.py --gt_path *** --pred_path ***
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
    thresholds = [i * 0.05 for i in range(20)]

    # compute metrics
    print('Computing full metrics...')
    results = compute_metrics(gt_images, pred_images, thresholds)
    display_f_score(results)
    ROC_curve(results, True)



