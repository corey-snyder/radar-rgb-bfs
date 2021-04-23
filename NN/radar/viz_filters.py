import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import IstaNet
import argparse
import yaml
from train import load_data
import matplotlib.pyplot as plt
from train import pad_mat
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
            plt.subplot(num_layers,num_weights,layer*num_weights+idx+1)
            try: plt.imshow(getattr(model.layers[layer],weight).weight[0,0].detach().numpy())
            except: plt.imshow(getattr(model.layers[layer],weight).weight[0].detach().numpy())
            plt.xticks([])
            plt.yticks([])
            if idx == 0: plt.ylabel('Layer ' + str(layer))
            plt.title(weight)
    # net_image=plt.imread('model_pic.png')
    # plt.subplot(num_layers+1,1,num_layers+1,)
    # plt.imshow(net_image)
    # plt.xticks([])
    # plt.yticks([])
    plt.show()


def view_sparse_progressions(full_rgb_input, full_radar_input, network, patch_shape, step_shape, device):
    # https://discuss.pytorch.org/t/how-to-split-tensors-with-overlap-and-then-reconstruct-the-original-tensor/70261
    torch.cuda.empty_cache()
    (height_step, width_step) = step_shape
    (patch_height,patch_width) = np.array(patch_shape[-2:])
    (batch_size,channels,og_height, og_width) = full_rgb_input.shape

    # determine padding (need to pad so that the size of the input is matches the patch/step size)
    padded_rgb_input, padded_radar_input = pad_mat(full_rgb_input, full_radar_input, (patch_height,patch_width),(height_step, width_step))

    patches_rgb = padded_rgb_input.unfold(2, size=int(patch_height), step=int(height_step)).unfold(3, size=int(patch_width), step=int(width_step))
    patches_radar = padded_radar_input.unfold(2, size=int(patch_width), step=int(width_step))
    patches_out = torch.zeros((batch_size, 2, patches_rgb.shape[2], patches_rgb.shape[3], patch_height, patch_width)).to(device)
    sparse_out = torch.zeros((batch_size, 2, patches_rgb.shape[2], patches_rgb.shape[3], patch_height, patch_width)).to(device)
    for ii in range(patches_rgb.shape[2]):
        for jj in range(patches_rgb.shape[3]):
            patch_rgb_input = patches_rgb[:,:,ii,jj].to(device)
            patch_radar_input = patches_radar[:,:,jj].to(device)
            L_patch_input = torch.zeros_like(patch_rgb_input).to(device)
            S_patch_input = torch.zeros_like(patch_rgb_input).to(device)
            patches_out[:,:,ii,jj], sparse_out[:,:,ii,jj] = network(patch_rgb_input, L_patch_input, S_patch_input, patch_radar_input,save_sparse=True)

    # fold data
    # reshape output to match F.fold input
    patches_out = patches_out.contiguous().view(batch_size, 2, -1, patch_height*patch_width)
    sparse_out = sparse_out.contiguous().view(batch_size, 2, -1, patch_height * patch_width)
    # print(patches_out.shape)  # [B, C, nb_patches_all, kernel_size*kernel_size]
    patches_out = patches_out.permute(0, 1, 3, 2)
    sparse_out = sparse_out.permute(0, 1, 3, 2)
    # print(patches_out.shape)  # [B, C, kernel_size*kernel_size, nb_patches_all]
    patches_out = patches_out.contiguous().view(batch_size, 2 * patch_height * patch_width, -1)
    sparse_out = sparse_out.contiguous().view(batch_size, 2 * patch_height * patch_width, -1)
    # print(patches_out.shape)  # [B, C*prod(kernel_size), L] as expected by Fold
    # https://pytorch.org/docs/stable/nn.html#torch.nn.Fold

    fold = torch.nn.Fold(output_size=padded_rgb_input.shape[-2:], kernel_size=(patch_height, patch_width), stride=(height_step, width_step))
    padded_output = fold(patches_out)
    padded_sparse_checkpoints = fold(sparse_out)

    # print(output.shape)  # [B, C, H, W]


    # fold ones
    ones_tensor = torch.ones_like(patches_out).to(device)

    # memory saver
    del L_patch_input, S_patch_input, patch_radar_input, patches_out, sparse_out
    torch.cuda.empty_cache()

    ones_tensor = fold(ones_tensor)

    # data /ones
    padded_output = padded_output / ones_tensor
    padded_sparse_checkpoints = padded_sparse_checkpoints / ones_tensor

    full_output = padded_output[:,:,:og_height,:og_width].to(device)
    full_sparse_checkpoints = padded_sparse_checkpoints[:,:,:og_height,:og_width].to(device)
    return full_output, full_sparse_checkpoints


def plot_data(full_rgb_input, full_radar_input, model, patch_shape, step_shape, frame, device):
    output, sparse_checkpoints = view_sparse_progressions(full_rgb_input, full_radar_input, model, patch_shape, step_shape, 'cpu')
    output, sparse_checkpoints = output.detach().numpy(), sparse_checkpoints.detach().numpy()
    fig = plt.figure(constrained_layout=True)
    spec2 = gridspec.GridSpec(ncols=2, nrows=len(model.layers)+1, figure=fig)
    for idx, layer in enumerate(model.layers):
        ax = fig.add_subplot(spec2[idx, 0])
        plt.imshow(sparse_checkpoints[frame,idx])
        plt.xticks([])
        plt.yticks([])
        plt.ylabel('Layer '+str(idx))
        plt.title('Before the Thresh')
        ax = fig.add_subplot(spec2[idx, 1])
        plt.plot((layer.lambda2*nn.functional.relu(layer.p7(full_radar_input))[frame,0]).detach().numpy())
        plt.xticks([])
        plt.title('Thresholds')
    ax = fig.add_subplot(spec2[-1, 0])
    plt.imshow(output[frame, 1])
    plt.xticks([])
    plt.yticks([])
    plt.title('Final Sparse Output')
    ax = fig.add_subplot(spec2[-1, 1])
    plt.plot(full_radar_input[frame,0].detach().numpy())
    plt.title('Radar Input')
    plt.xticks([])

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

    plot_data(D[:30], R[:30], model, data_shape, [30, 30], 10, 'cpu')

    plot_weights(model)