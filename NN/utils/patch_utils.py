import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from skimage.transform import rescale
import argparse
import yaml
from datetime import datetime
import shutil


def random_crop(D, target, patch_shape, R=None):
    (full_height, full_width) = D.shape[-2:]
    (patch_height, patch_width) = patch_shape

    # assuming coordinates belong to center of the patch so we have to be side_len/2 away from boundary
    height_range = (patch_height // 2, full_height - patch_height // 2)
    width_range = (patch_width // 2, full_width - patch_width // 2)

    rand_height = np.random.choice(np.arange(height_range[0], height_range[1]))
    rand_width = np.random.choice(np.arange(width_range[0], width_range[1]))

    # Extracting Patch
    D_patch = D[:, :, rand_height - patch_height // 2: rand_height + patch_height // 2,
              rand_width - patch_width // 2: rand_width + patch_width // 2]
    target_patch = target[:, :, rand_height - patch_height // 2:rand_height + patch_height // 2,
                   rand_width - patch_width // 2:rand_width + patch_width // 2]

    if R is not None:
        R_patch = R[:, :, rand_width - patch_width // 2: rand_width + patch_width // 2]
        return D_patch, R_patch, target_patch
    return D_patch, target_patch


def pad_mat(og_input, patch_shape, step_shape):
    (_, _, og_height, og_width) = og_input.shape
    (height_step, width_step) = step_shape
    (patch_height, patch_width) = patch_shape

    # determine height padding amount
    new_height = patch_height
    while new_height < og_height:
        new_height += height_step

    # determine width padding amount
    new_width = patch_width
    while new_width < og_width:
        new_width += width_step

    height_pad_len = new_height - og_height
    width_pad_len = new_width - og_width

    pad = torch.nn.ZeroPad2d((0, width_pad_len, 0, height_pad_len))

    padded_input = pad(og_input)

    return padded_input


def infer_full_image(full_input, network, patch_shape, step_shape, device):
    # https://discuss.pytorch.org/t/how-to-split-tensors-with-overlap-and-then-reconstruct-the-original-tensor/70261
    torch.cuda.empty_cache()
    (height_step, width_step) = step_shape
    (patch_height, patch_width) = np.array(patch_shape[-2:])
    (batch_size, channels, og_height, og_width) = full_input.shape

    # determine padding (need to pad so that the size of the input is matches the patch/step size)
    padded_input = pad_mat(full_input, (patch_height, patch_width), (height_step, width_step))

    patches = padded_input.unfold(2, size=int(patch_height), step=int(height_step)).unfold(3, size=int(patch_width), step=int(width_step))
    patches_out = torch.zeros((batch_size, 2, patches.shape[2], patches.shape[3], patch_height, patch_width)).to(device)
    for ii in range(patches.shape[2]):
        for jj in range(patches.shape[3]):
            patch_input = patches[:, :, ii, jj].to(device)
            patches_out[:, :, ii, jj] = network(patch_input)

    # fold data
    # reshape output to match F.fold input
    patches_out = patches_out.contiguous().view(batch_size, 2, -1, patch_height * patch_width)
    # print(patches_out.shape)  # [B, C, nb_patches_all, kernel_size*kernel_size]
    patches_out = patches_out.permute(0, 1, 3, 2)
    # print(patches_out.shape)  # [B, C, kernel_size*kernel_size, nb_patches_all]
    patches_out = patches_out.contiguous().view(batch_size, 2 * patch_height * patch_width, -1)
    # print(patches_out.shape)  # [B, C*prod(kernel_size), L] as expected by Fold
    # https://pytorch.org/docs/stable/nn.html#torch.nn.Fold

    fold = torch.nn.Fold(output_size=padded_input.shape[-2:], kernel_size=(patch_height, patch_width), stride=(height_step, width_step))
    padded_output = fold(patches_out)
    # print(output.shape)  # [B, C, H, W]

    # fold ones
    ones_tensor = torch.ones_like(patches_out).to(device)

    # memory saver
    del patches_out
    torch.cuda.empty_cache()

    ones_tensor = fold(ones_tensor)

    # data /ones
    padded_output = padded_output / ones_tensor
    full_output = padded_output[:, :, :og_height, :og_width].to(device)

    return full_output


def pad_mat_radar(og_rgb_input, og_radar_input, patch_shape, step_shape):
    (_, _, og_height, og_width) = og_rgb_input.shape
    (height_step, width_step) = step_shape
    (patch_height, patch_width) = patch_shape

    # determine height padding amount
    new_height = patch_height
    while new_height < og_height:
        new_height += height_step

    # determine width padding amount
    new_width = patch_width
    while new_width < og_width:
        new_width += width_step

    height_pad_len = new_height-og_height
    width_pad_len = new_width-og_width

    pad_2d = torch.nn.ZeroPad2d((0,width_pad_len,0, height_pad_len))
    pad_1d = torch.nn.ConstantPad1d(width_pad_len,0)

    padded_rgb_input = pad_2d(og_rgb_input)
    padded_radar_input = pad_1d(og_radar_input)

    return padded_rgb_input, padded_radar_input


def infer_full_image_radar(full_rgb_input, full_radar_input, network, patch_shape, step_shape, device):
    # https://discuss.pytorch.org/t/how-to-split-tensors-with-overlap-and-then-reconstruct-the-original-tensor/70261
    torch.cuda.empty_cache()
    (height_step, width_step) = step_shape
    (patch_height,patch_width) = np.array(patch_shape[-2:])
    (batch_size,channels,og_height, og_width) = full_rgb_input.shape

    # determine padding (need to pad so that the size of the input is matches the patch/step size)
    padded_rgb_input, padded_radar_input = pad_mat_radar(full_rgb_input, full_radar_input, (patch_height,patch_width),(height_step, width_step))

    patches_rgb = padded_rgb_input.unfold(2, size=int(patch_height), step=int(height_step)).unfold(3, size=int(patch_width), step=int(width_step))
    patches_radar = padded_radar_input.unfold(2, size=int(patch_width), step=int(width_step))
    patches_out = torch.zeros((batch_size, 2, patches_rgb.shape[2], patches_rgb.shape[3], patch_height, patch_width)).to(device)
    for ii in range(patches_rgb.shape[2]):
        for jj in range(patches_rgb.shape[3]):
            patch_rgb_input = patches_rgb[:,:,ii,jj].to(device)
            patch_radar_input = patches_radar[:,:,jj].to(device)
            # L_patch_input = torch.zeros_like(patch_rgb_input).to(device)
            # S_patch_input = torch.zeros_like(patch_rgb_input).to(device)
            patches_out[:,:,ii,jj] = network(patch_rgb_input, patch_radar_input)

    # fold data
    # reshape output to match F.fold input
    patches_out = patches_out.contiguous().view(batch_size, 2, -1, patch_height*patch_width)
    # print(patches_out.shape)  # [B, C, nb_patches_all, kernel_size*kernel_size]
    patches_out = patches_out.permute(0, 1, 3, 2)
    # print(patches_out.shape)  # [B, C, kernel_size*kernel_size, nb_patches_all]
    patches_out = patches_out.contiguous().view(batch_size, 2 * patch_height * patch_width, -1)
    # print(patches_out.shape)  # [B, C*prod(kernel_size), L] as expected by Fold
    # https://pytorch.org/docs/stable/nn.html#torch.nn.Fold

    fold = torch.nn.Fold(output_size=padded_rgb_input.shape[-2:], kernel_size=(patch_height, patch_width), stride=(height_step, width_step))
    padded_output = fold(patches_out)
    # print(output.shape)  # [B, C, H, W]


    # fold ones
    ones_tensor = torch.ones_like(patches_out).to(device)

    # memory saver
    del patch_radar_input, patches_out
    torch.cuda.empty_cache()

    ones_tensor = fold(ones_tensor)

    # data /ones
    padded_output = padded_output / ones_tensor
    full_output = padded_output[:,:,:og_height,:og_width].to(device)

    return full_output
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from data_loader import *

    idx = 13

    D, L, S, R = load_radar_rgb_data('../../Data/CSL_lobby_side_0/raw',50,.20,400,False)
    target = torch.cat((L, S),1)  # concatenate the L and S in the channel dimension

    D_patch, R_patch, target_patch = random_crop(D,target,[80,80],R)
    D_patch, R_patch, target_patch = D_patch.detach().numpy(), R_patch.detach().numpy(), target_patch.detach().numpy()
    plt.figure()
    plt.subplot(141)
    plt.imshow(D_patch[idx,0])
    plt.subplot(142)
    plt.imshow(target_patch[idx, 0])
    plt.subplot(143)
    plt.imshow(target_patch[idx, 1])
    plt.subplot(144)
    plt.plot(R_patch[idx,0])
    plt.show()
