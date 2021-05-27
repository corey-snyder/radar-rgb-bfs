import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import yaml
from datetime import datetime
import shutil
import matplotlib.pyplot as plt
from unet.unet_model import UNet
from utils.data_loader import *
from utils.patch_utils import *
from utils.tensorboard_utils import *


if __name__ == '__main__':
    # Get command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-yaml", help="path of yaml file", type=str)
    args = parser.parse_args()
    yaml_path = args.yaml

    # Load yaml and get contents
    with open(yaml_path) as file:
        setup_dict = yaml.load(file,Loader=yaml.FullLoader)

    train_path = setup_dict['train_path']
    test_path = setup_dict['test_path']
    try_gpu = setup_dict['try_gpu']
    downsample_rate = setup_dict['downsample']  # in each dim
    learning_rate = setup_dict['lr']
    schedule_step = setup_dict['schedule_step']
    schedule_multiplier = setup_dict['schedule_multiplier']  # <1
    patch_height = setup_dict['patch_height']
    patch_width = setup_dict['patch_width']
    seed = setup_dict['seed']
    seq_len = setup_dict['seq_len']
    thresh = setup_dict['thresh']

    # set seeds
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
    D_train_full, _, S_train_full_target = load_rgb_data(train_path, seq_len, rescale_factor=downsample_rate)
    D_train_full.to(device)

    D_test_full, _, S_test_full_target = load_rgb_data(test_path, seq_len, rescale_factor=downsample_rate)
    D_test_full.to(device)

    # Threshold Data
    S_train_full_target = threshold_data(S_train_full_target, thresh)
    S_test_full_target = threshold_data(S_test_full_target, thresh)

    # Swap axes (1 image with many channels)
    D_train_full, D_test_full = torch.transpose(D_train_full,0,1), torch.transpose(D_test_full,0,1)
    S_train_full_target, S_test_full_target = torch.transpose(S_train_full_target,0,1), torch.transpose(S_test_full_target,0,1)

    # Destination for tensorboard log data
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    log_name = 'UNET_thresh' + str(thresh) + '_lr' + str(learning_rate) + '_ds' + str(downsample_rate) + '_seed' + str(seed) + '_len' + str(seq_len)
    log_dir = './runs/' + log_name + '__' + dt_string
    writer = SummaryWriter(log_dir)
    shutil.copyfile(yaml_path, log_dir + '/train.yaml')

    # init model
    data_shape = list(np.concatenate([D_train_full.shape[:2], [patch_height, patch_width]]))
    model = UNet(n_channels=seq_len, n_classes=seq_len)
    model.to(device)
    # specify loss function
    criterion = nn.BCELoss()
    # specify optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=schedule_multiplier, step_size=schedule_step)

    # tensorboard graph
    writer.add_graph(model, D_train_full[:, :, :patch_height, :patch_width].to(device))

    n_epochs = 50000  # number of epochs to train the model
    valid_loss_min = np.Inf  # track change in validation loss
    train_loss_min = np.Inf  # track change in train loss


    #################################################################

    for epoch in range(1, n_epochs + 1):

        ###################
        # train the model #
        ###################

        model.train()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()

        # get random crop
        D_train_patch, S_train_patch_target = random_crop(D_train_full, S_train_full_target, (patch_height, patch_width))
        # send tensors to device
        D_train_patch, S_train_patch_target = D_train_patch.to(device),  S_train_patch_target.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(D_train_patch)
        # calculate the batch loss
        loss = criterion(output, S_train_patch_target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()

        ######################
        # validate the model #
        ######################

        if epoch % 250 == 0:
            # memory saver
            del loss

            model.eval()
            step_shape = np.array(data_shape[-2:])//2
            output_train_full = model(D_train_full.to(device))

            # compute pixel loss of entire training image sequence
            S_train_full_target.to(device)
            loss = criterion(output_train_full,S_train_full_target.to(device))
            train_loss = loss.item()
            writer.add_scalar('Training Avg. Pixel BCE Loss',
                              loss.item(),
                              epoch)

            # if (epoch % 250 ==0) and (epoch < 10000):
            #    writer.add_figure('Predictions vs. Targets TRAIN',
            #              plot_classes_preds_unet(output_train_full.cpu().detach().numpy(),D_train_full.cpu().numpy(),S_train_full_target.cpu().numpy()),
            #              global_step=epoch)

            del output_train_full, loss

            # compute pixel loss of entire testing image sequence
            output_test_full = model(D_test_full.to(device))
            loss = criterion(output_test_full, S_test_full_target.to(device))
            valid_loss = loss.item()
            writer.add_scalar('Testing Avg. Pixel BCE Loss',
                              loss.item(),
                              epoch)

            # if (epoch % 250 ==0) and (epoch < 10000):
            #    writer.add_figure('Predictions vs. Targets TEST',
            #                  plot_classes_preds_unet(output_test_full.cpu().detach().numpy(), D_test_full.cpu().numpy(),
            #                                     S_test_full_target.cpu().numpy()),
            #                  global_step=epoch)

            del output_test_full, loss

            # print('Epoch: {} \tTraining Loss: {:.6f} \tTest Loss: {:.6f}'.format(epoch, train_loss, valid_loss))

            # save model if test loss has decreased
            if (valid_loss <= valid_loss_min) and (valid_loss != 0):
                print('Epoch: {} \tValidation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(epoch, valid_loss_min, valid_loss))
                torch.save(model.state_dict(), log_dir + '/model_bfs_test.pt')
                valid_loss_min = valid_loss

            if (train_loss <= train_loss_min) and (train_loss != 0):
                print('Epoch: {} \tTrain loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(epoch, train_loss_min, train_loss))
                torch.save(model.state_dict(), log_dir + '/model_bfs_train.pt')
                train_loss_min = train_loss

        scheduler.step()
