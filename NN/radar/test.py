import numpy as np
import matplotlib.pyplot as plt
import torch
from model import IstaNet
from train import load_data
import argparse
import yaml


def plot_func(output,D,R):
    n = 3
    N = len(D) // n
    R_min = np.min(R)
    R_max = np.max(R)

    plt.figure(figsize=(7, 35))
    for ii in range(N):
        plt.subplot(N, 4, 4 * ii + 1)
        plt.imshow(D[n * ii, 0], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if ii == 0: plt.title('Original')

        plt.subplot(N, 4, 4 * ii + 2)
        plt.imshow(output[n * ii, 0], cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if ii == 0: plt.title('L')

        plt.subplot(N, 4, 4 * ii + 3)
        plt.imshow(np.abs(output[n * ii, 1]), cmap='gray')
        plt.xticks([])
        plt.yticks([])
        if ii == 0: plt.title('abs(S)')

        plt.subplot(N, 4, 4 * ii + 4)
        plt.plot(R[n * ii,0])
        plt.ylim([R_min,R_max])
        plt.xticks([])
        if ii == 0: plt.title('Radar')
    plt.tight_layout()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-yaml", help="path of yaml file", type=str)
    args = parser.parse_args()
    yampl_path = args.yaml

    with open(yampl_path) as file:
        setup_dict = yaml.load(file, Loader=yaml.FullLoader)
    run_path = setup_dict['run_path'][0]
    test_path = setup_dict['test_path'][0]
    radar_data_test = setup_dict['test_radar_input'][0]
    try_gpu = setup_dict['try_gpu'][0]

    net_path = run_path + '/model_bfs.pt'
    yaml_train_path = run_path + '/setup.yaml'

    with open(yaml_train_path) as file:
        setup_dict = yaml.load(file, Loader=yaml.FullLoader)
    n_layers = setup_dict['n_layers'][0]
    downsample_rate = setup_dict['downsample'][0]

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

    D, L_target, S_target, R = load_data(test_path, radar_data=radar_data_test, rescale_factor=downsample_rate)
    L_test = torch.zeros_like(D)
    S_test = torch.zeros_like(D)
    D, L_test, S_test, R = D.to(device), L_test.to(device), S_test.to(device), R.to(device)

    (n_frames,n_channels,im_height,im_width) = D.shape
    model = IstaNet(D.shape,n_layers)
    model.load_state_dict(torch.load(net_path))
    model.to(device)
    model.eval()

    output = model(D,L_test,S_test,R)
    output, D, R = output.detach().cpu().numpy(), D.detach().cpu().numpy(), R.detach().cpu().numpy()
    plot_func(output,D,R)
    plt.show()
