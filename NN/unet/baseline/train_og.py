import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from unet_model import UNet
from torch.utils.tensorboard import SummaryWriter

from skimage.transform import rescale
import argparse
import yaml
from datetime import datetime
import shutil
from sys import getsizeof

import matplotlib.pyplot as plt


def load_data(path, threshold, n_frames=30, rescale_factor=1., ):
    D = np.load(path + '/D.npy')
    S = np.load(path + '/S_pcp.npy')

    # Add channel dimension, new shape = [n_frames,1,720,1280] not including downsampling
    D = D[:n_frames, None, :, :]
    S = S[:n_frames, None, :, :]

    # Downsample with anti-aliasing filter
    if rescale_factor != 1:
        D = rescale(D, (1, 1, rescale_factor, rescale_factor), anti_aliasing=True)
        S = rescale(S, (1, 1, rescale_factor, rescale_factor), anti_aliasing=True)

    # Threshold the Sparse component
    S = np.abs(S)
    S[S < threshold] = 0
    S[S >= threshold] = 1
    print('Thresholding the Sparse Component at ', threshold)
    return torch.from_numpy(D).float(), torch.from_numpy(S).float()


def random_crop(D, target, patch_shape):
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

    return D_patch, target_patch


def pad_mat(og_input,patch_shape, step_shape):
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

    height_pad_len = new_height-og_height
    width_pad_len = new_width-og_width

    pad = torch.nn.ZeroPad2d((0,width_pad_len,0, height_pad_len))

    padded_input = pad(og_input)

    return padded_input


def infer_full_image(full_input, network, patch_shape, step_shape, device):
    # https://discuss.pytorch.org/t/how-to-split-tensors-with-overlap-and-then-reconstruct-the-original-tensor/70261
    torch.cuda.empty_cache()
    (height_step, width_step) = step_shape
    (patch_height,patch_width) = np.array(patch_shape[-2:])
    (batch_size,channels,og_height, og_width) = full_input.shape

    # determine padding (need to pad so that the size of the input is matches the patch/step size)
    padded_input = pad_mat(full_input,(patch_height,patch_width),(height_step, width_step))

    patches = padded_input.unfold(2, size=int(patch_height), step=int(height_step)).unfold(3, size=int(patch_width), step=int(width_step))
    patches_out = torch.zeros((batch_size, 2, patches.shape[2], patches.shape[3], patch_height, patch_width)).to(device)
    for ii in range(patches.shape[2]):
        for jj in range(patches.shape[3]):
            patch_input = patches[:,:,ii,jj].to(device)
            L_patch_input = torch.zeros_like(patch_input).to(device)
            S_patch_input = torch.zeros_like(patch_input).to(device)
            patches_out[:,:,ii,jj] = network(patch_input, L_patch_input, S_patch_input)

    # fold data
    # reshape output to match F.fold input
    patches_out = patches_out.contiguous().view(batch_size, 2, -1, patch_height*patch_width)
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
    del L_patch_input, S_patch_input, patches_out
    torch.cuda.empty_cache()


    ones_tensor = fold(ones_tensor)

    # data /ones
    padded_output = padded_output / ones_tensor
    full_output = padded_output[:,:,:og_height,:og_width].to(device)

    return full_output

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-yaml", help="path of yaml file", type=str)
    args = parser.parse_args()
    yampl_path = args.yaml

    with open(yampl_path) as file:
        setup_dict = yaml.load(file, Loader=yaml.FullLoader)
    train_path = setup_dict['train_path'][0]
    test_path = setup_dict['test_path'][0]
    try_gpu = setup_dict['try_gpu'][0]
    downsample_rate = setup_dict['downsample'][0]  # in each dim
    learning_rate = setup_dict['lr'][0]
    schedule_step = setup_dict['schedule_step'][0]
    schedule_multiplier = setup_dict['schedule_multiplier'][0]  # <1
    patch_height = setup_dict['patch_height'][0]
    patch_width = setup_dict['patch_width'][0]
    threshold = setup_dict['threshold'][0]
    seed = setup_dict['seed'][0]

    torch.manual_seed(seed)
    np.random.seed(seed)

    # check if CUDA is available
    if try_gpu:
        train_on_gpu = torch.cuda.is_available()
    else:
        train_on_gpu = False

    device = torch.device("cuda" if train_on_gpu else "cpu")
    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')

    # load Data (not dataset object since no train/test split)
    D_train_full, S_train_full = load_data(train_path, threshold=threshold, rescale_factor=downsample_rate)
    data_shape = list(np.concatenate([D_train_full.shape[:2],[patch_height,patch_width]]))
    #print(D_train_full.element_size()*D_train_full.nelement())
    #print(getsizeof(S_train_full))
    print("allocated ",torch.cuda.memory_allocated(0))
    D_test_full, S_test_full = load_data(test_path, threshold = threshold, rescale_factor=downsample_rate)
    # Destination for tensorboard log data
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    log_name = 'UNET_OG_thresh_' + str(threshold) + '_lr_' + str(learning_rate) + '_ds_' + str(downsample_rate)
    log_dir = '../runs/' + log_name + '__' + dt_string
    writer = SummaryWriter(log_dir)
    shutil.copyfile(yampl_path, log_dir + '/setup.yaml')

    # init model
    model = UNet(n_channels=1, n_classes=1)
    model.to(device)
    # specify loss function (categorical cross-entropy)
    criterion = nn.BCELoss()
    # specify optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=schedule_multiplier, step_size=schedule_step)

    # tensorboard graph
    #writer.add_graph(model, (D_train_full[:, :, :patch_height, :patch_width].to(device), D_train_full[:, :, :patch_height, :patch_width].to(device),
    #                         D_train_full[:, :, :patch_height, :patch_width].to(device)))
    # writer.close()

    # train

    n_epochs = 50000  # number of epochs to train the model
    valid_loss_min = np.Inf  # track change in validation loss

    for epoch in range(1, n_epochs + 1):

        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        model.train()
        model.to(device)
        print("allocated ",torch.cuda.memory_allocated(0))
        # clear the gradients of all optimized variables
        optimizer.zero_grad()

        # get random crop
        D_train_patch, S_train_patch = random_crop(D_train_full, S_train_full, (patch_height, patch_width))
        # send tensors to device
        D_train_patch, S_train_patch = D_train_patch.to(device), S_train_patch.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(D_train_patch)
        # calculate the batch loss
        loss = criterion(output, S_train_patch)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()

        # add training loss in tensorboard from patch
        # if epoch % 10 == 0:
        #     writer.add_scalar('Training Pixel loss Patch',
        #                   loss.item(),
        #                   epoch)
        #     writer.close()

        # add values of singular values of L in tensorboard
        # (n_frames, n_channels, im_height, im_width) = D_train_patch.shape
        # L_flat = torch.reshape(output[:,0], (-1, patch_height * patch_width)).T
        # (u, s, v) = torch.svd(L_flat)
        # # s_dict = {}
        # # for idx, ii in enumerate(s):
        # #     s_dict[str(idx)] = s[idx].item()
        # # writer.add_scalars('sing. vals of L', s_dict,epoch)
        # # writer.close()
        #
        # # add rank of L to tensorboard
        # writer.add_scalar('rank of L', sum(s>1e-4),epoch)
        # writer.close()

        # show sample predictions
        if epoch % 10 == 0:
            # memory saver

            del S_train_patch, D_train_patch, output, loss
            torch.cuda.empty_cache()
            step_shape = np.array(data_shape[-2:])//2
            model.eval()
            #print(D_train_full.shape)
            #print(getsizeof(D_train_full))
            #output_train_full = model.to('cpu')(D_train_full)
            #D_train_full = D_train_full.to(device)
            #print("allocated ",torch.cuda.memory_allocated(0))
            #output_train_full = model(D_train_full)
            output_train_full = infer_full_image(D_train_full,model,data_shape,step_shape, device)
            plt.imshow(output_train_full[15, 0].detach().cpu().numpy())
            plt.show()
            # compute pixel loss of entire image
            loss = criterion(output_train_full, S_train_full)
            train_loss = loss.item()
            writer.add_scalar('Training Pixel loss Full',
                             loss.item(),
                             epoch)

            # if (epoch % 1000 ==0) and (epoch <10000):
               # writer.add_figure('predictions vs. actuals TRAIN',
               #           plot_classes_preds(output_train_full.cpu().detach().numpy(),L_train_full.cpu().numpy(),S_train_full.cpu().numpy()),
               #           global_step=epoch)

            del output_train_full, loss

            ######################
            # validate the model #
            ######################
		
            #output_test_full = model.to('cpu')(D_test_full)
            output_test_full = model(D_test_full.to(device))
            plt.imshow(output_test_full[5,0].detach().cpu().numpy())
            plt.show()
            loss = criterion(output_test_full, S_test_full)
            valid_loss = loss.item()
            writer.add_scalar('Testing Pixel loss Full',
                              loss.item(),
                              epoch)
            # writer.close()
            # if (epoch % 1000 ==0) and (epoch <10000):
            #    writer.add_figure('predictions vs. actuals TEST',
            #                  plot_classes_preds(output_test_full.cpu().detach().numpy(), L_test_full_target.cpu().numpy(),
            #                                     S_test_full_target.cpu().numpy()),
            #                  global_step=epoch)
            # writer.close()

            print('Epoch: {} \tTraining Loss: {:.6f} \tTest Loss: {:.6f}'.format(
                epoch, train_loss, valid_loss))

            # save model if validation loss has decreased
            if (valid_loss <= valid_loss_min) and (valid_loss != 0):
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    valid_loss_min,
                    valid_loss))
                torch.save(model.state_dict(), log_dir + '/model_bfs.pt')
                valid_loss_min = valid_loss

            del output_test_full, loss

        else:
            print('Epoch: {} \tTraining Loss: {:.6f}'.format(
                epoch, train_loss))

        scheduler.step()