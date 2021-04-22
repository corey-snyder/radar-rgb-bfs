import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import IstaNet
import argparse
import yaml
from train import load_data, infer_full_image
import matplotlib.pyplot as plt
from matplotlib import gridspec


from datetime import datetime
import shutil


def plot_weights(model):
    weight_names = ['p1','p2','p3','p4','p5','p6','p7']
    num_layers = len(model.layers)
    plt.figure()
    for layer in range(num_layers):
        num_weights = len(weight_names)
        for idx, weight in enumerate(weight_names):
            plt.subplot(num_layers+2,num_weights,layer*num_weights+idx+1)
            try: plt.imshow(getattr(model.layers[layer],weight).weight[0,0].detach().numpy())
            except: plt.imshow(getattr(model.layers[layer],weight).weight[0].detach().numpy())
            plt.xticks([])
            plt.yticks([])
            if idx == 0: plt.ylabel('Layer ' + str(layer))
            plt.title(weight)
    net_image=plt.imread('model_pic.png')
    plt.subplot(num_layers+1,1,num_layers+1,)
    plt.imshow(net_image)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()



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

    net_path = run_path + '/model_bfs.pt'
    yaml_train_path = run_path + '/setup.yaml'

    with open(yaml_train_path) as file:
        setup_dict = yaml.load(file, Loader=yaml.FullLoader)
    n_layers = setup_dict['n_layers'][0]
    downsample_rate = setup_dict['downsample'][0]
    patch_height = setup_dict['patch_height'][0]
    patch_width = setup_dict['patch_width'][0]

    D, L_target, S_target, R = load_data(test_path, radar_data=radar_data_test, rescale_factor=downsample_rate)
    data_shape = list(np.concatenate([D.shape[:2], [patch_height, patch_width]]))

    model = IstaNet(data_shape, n_layers)
    model.load_state_dict(torch.load(net_path))
    model.eval()

    # output = infer_full_image(D[:30], R[:30], model, data_shape, [30,30], 'cpu')
    # plt.imshow(np.abs(output[10,1].detach().numpy()))
    # plt.show()

    plot_weights(model)
