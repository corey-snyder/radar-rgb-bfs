import numpy as np
import matplotlib.pyplot as plt
import torch
from model import IstaNet
from train import load_data
import argparse
import yaml
from train import infer_full_image
from test import plot_func
import os
from skimage.io import imread
import sys
sys.path.append('/mnt/data0-nfs/cesnyde2/radar-rgb-bfs')
from evaluate import compute_metrics, display_results, ROC_curve

def pad_or_truncate(some_list, target_len):
    return np.concatenate([some_list,np.array(np.nan*np.ones(target_len - len(some_list)))])

def display_dir_results(results_list):
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
    auc_mean = np.mean(all_AUCS)
    auc_std = np.std(all_AUCS)
    #
    # max_len = 0
    # for FPRS in all_FPRS:
    #     if len(FPRS) > max_len:
    #         max_len = len(FPRS)
    #         max_FPRS = FPRS
    #
    # all_TPRS = np.array([pad_or_truncate(ii, max_len) for ii in all_TPRS])
    #
    # TPRS_means = np.nanmean(all_TPRS, 0)

    # plt.plot(max_FPRS, TPRS_means)
    plt.title('ROC Curves AUC = N({:.3f},{:.3f})'.format(np.mean(all_AUCS),np.var(all_AUCS)))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

    return auc_mean, auc_std


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-yaml", help="path of yaml file", type=str)
    args = parser.parse_args()
    yaml_test_path = args.yaml

    with open(yaml_test_path) as file:
        setup_dict = yaml.load(file, Loader=yaml.FullLoader)
    runs_dir = setup_dict['runs_dir'][0]
    test_data = setup_dict['test_data'][0]
    test_radar_data = setup_dict['test_radar_data'][0]
    try_gpu = setup_dict['try_gpu'][0]
    step_height = setup_dict['step_height'][0]
    step_width = setup_dict['step_width'][0]
    gt_path = setup_dict['GT'][0]

    runs = os.listdir(runs_dir)
    results = []
    # set up for F-scores
    thresholds = [i * 0.05 for i in range(20)]
    filenames = os.listdir(gt_path)
    gt_images = []
    for ii in range(len(filenames)):
        gt_images.append(imread(os.path.join(gt_path, 'rgb_%i.png' % ii)))

    for run in runs:
        run_path = runs_dir + run
        print('\n'+run_path)
        net_path = run_path + '/model_bfs.pt'
        yaml_train_path = run_path + '/setup.yaml'

        with open(yaml_train_path) as file:
            setup_dict = yaml.load(file, Loader=yaml.FullLoader)
        n_layers = setup_dict['n_layers'][0]
        downsample_rate = setup_dict['downsample'][0]
        patch_height = setup_dict['patch_height'][0]
        patch_width = setup_dict['patch_width'][0]

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

        D, L_target, S_target = load_data(test_data, rescale_factor=downsample_rate, print_flag=False)
        data_shape = list(np.concatenate([D.shape[:2],[patch_height,patch_width]]))
        step_shape = (step_height, step_width)

        (n_frames,n_channels,im_height,im_width) = D.shape
        model = IstaNet(data_shape,n_layers)
        model.load_state_dict(torch.load(net_path))
        model.to(device)
        model.eval()

        output = infer_full_image(D, model, data_shape, step_shape, device)
        output, D = output.detach().cpu().numpy(), D.detach().cpu().numpy()

        # Compute F-scores
        if downsample_rate !=1:
            gt_images_temp = [gt_images[ii][::int(1/downsample_rate),::int(1/downsample_rate)] for ii in range(len(gt_images))]
        output_len = output.shape[0]
        gt_images_temp = gt_images_temp[:output_len]
        pred_images = [np.array(np.abs(output[ii,1])*255,dtype=np.uint8) for ii in range(output_len)]  # max abs val of float image is 1

        print('Computing full metrics...')
        temp_results = compute_metrics(gt_images_temp, pred_images, thresholds, False)
        # display_results(temp_results)
        results.append(temp_results)
    print('\n')
    auc_mean,auc_std = display_dir_results(results)
    print('Mean AUC = {:.5f} +/- {:.5f}'.format(auc_mean, auc_std))
    #np.save('roc_auc/l8_radar_auc_csl_pantry_12.npy',[ auc_mean,auc_std],allow_pickle=True)
