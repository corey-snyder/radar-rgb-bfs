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
        if epoch % 100 == 0:
            # memory saver

            del S_train_patch, D_train_patch, output, loss
            torch.cuda.empty_cache()
            model.eval()

            output_train_full = model.to('cpu')(D_train_full)
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

            output_test_full = model.to('cpu')(D_test_full)
            plt.imshow(output_test_full[5,0].detach().cpu().numpy())
            plt.show()
            loss = criterion(output_test_full, S_test_full)
            valid_loss = loss.item()
            writer.add_scalar('Testing Pixel loss Full',
                              loss.item(),
                              epoch)
            writer.close()
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
                # torch.save(model.state_dict(), log_dir + '/model_bfs.pt')
                valid_loss_min = valid_loss

            del output_test_full, loss

        else:
            print('Epoch: {} \tTraining Loss: {:.6f}'.format(
                epoch, train_loss))

        scheduler.step()
