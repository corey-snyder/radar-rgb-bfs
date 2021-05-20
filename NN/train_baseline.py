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
from baseline.model import IstaNet
from utils.data_loader import *
from utils.patch_utils import *
from utils.tensorboard_utils import plot_classes_preds


if __name__ == '__main__':
    # Get command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-yaml", help="path of yaml file", type=str)
    args = parser.parse_args()
    yampl_path = args.yaml

    # Load yaml and get contents
    with open(yampl_path) as file:
        setup_dict = yaml.load(file,Loader=yaml.FullLoader)

    train_path = setup_dict['train_path']
    test_path = setup_dict['test_path']
    n_layers = setup_dict['n_layers']
    try_gpu = setup_dict['try_gpu']
    downsample_rate = setup_dict['downsample']  # in each dim
    learning_rate = setup_dict['lr']
    schedule_step = setup_dict['schedule_step']
    schedule_multiplier = setup_dict['schedule_multiplier']  # <1
    patch_height = setup_dict['patch_height']
    patch_width = setup_dict['patch_width']
    seed = setup_dict['seed']
    seq_len = setup_dict['seq_len']

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
    D_train_full, L_train_full_target, S_train_full_target = load_rgb_data(train_path, seq_len, rescale_factor=downsample_rate)
    target_train_full = torch.cat((L_train_full_target, S_train_full_target), 1)  # concatenate the L and S in the channel dimension
    D_train_full.to(device)

    D_test_full, L_test_full_target, S_test_full_target = load_rgb_data(test_path, seq_len, rescale_factor=downsample_rate)
    target_test_full = torch.cat((L_test_full_target, S_test_full_target), 1)  # concatenate the L and S in the channel dimension
    D_test_full.to(device)

    # Destination for tensorboard log data
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    log_name = 'BASELINE_l' + str(n_layers) + '_lr' + str(learning_rate) + '_ds' + str(downsample_rate) + '_seed' + str(seed)
    log_dir = './runs/' + log_name + '__' + dt_string
    writer = SummaryWriter(log_dir)
    shutil.copyfile(yampl_path, log_dir + '/train.yaml')

    # init model
    data_shape = list(np.concatenate([D_train_full.shape[:2], [patch_height, patch_width]]))
    model = IstaNet(data_shape, n_layers)
    model.to(device)
    # specify loss function
    criterion = nn.MSELoss()
    # specify optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=schedule_multiplier, step_size=schedule_step)

    # tensorboard graph
    writer.add_graph(model, D_train_full[:, :, :patch_height, :patch_width].to(device))

    n_epochs = 50000  # number of epochs to train the model
    valid_loss_min = np.Inf  # track change in validation loss

    #################################################################

    for epoch in range(1, n_epochs + 1):

        ###################
        # train the model #
        ###################

        model.train()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()

        # get random crop
        D_train_patch, target_train_patch = random_crop(D_train_full, target_train_full, (patch_height, patch_width))
        # send tensors to device
        D_train_patch, target_train_patch = D_train_patch.to(device),  target_train_patch.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(D_train_patch)
        # calculate the batch loss
        loss = criterion(output, target_train_patch)
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
            output_train_full = infer_full_image(D_train_full,model,data_shape,step_shape, device)

            # compute pixel loss of entire training image sequence
            target_train_full.to(device)
            loss = criterion(output_train_full,target_train_full.to(device))
            train_loss = loss.item()
            writer.add_scalar('Training Avg. Pixel MSE Loss',
                              loss.item(),
                              epoch)

            if (epoch % 1000 ==0) and (epoch < 10000):
               writer.add_figure('Predictions vs. Targets TRAIN',
                         plot_classes_preds(output_train_full.cpu().detach().numpy(),L_train_full_target.cpu().numpy(),S_train_full_target.cpu().numpy()),
                         global_step=epoch)

            del output_train_full, loss

            # compute pixel loss of entire testing image sequence
            output_test_full = infer_full_image(D_test_full, model, data_shape, step_shape, device)
            loss = criterion(output_test_full, target_test_full.to(device))
            valid_loss = loss.item()
            writer.add_scalar('Testing Avg. Pixel MSE Loss',
                              loss.item(),
                              epoch)

            if (epoch % 1000 ==0) and (epoch < 10000):
               writer.add_figure('Predictions vs. Targets TEST',
                             plot_classes_preds(output_test_full.cpu().detach().numpy(), L_test_full_target.cpu().numpy(),
                                                S_test_full_target.cpu().numpy()),
                             global_step=epoch)

            del output_test_full, loss

            # print('Epoch: {} \tTraining Loss: {:.6f} \tTest Loss: {:.6f}'.format(epoch, train_loss, valid_loss))

            # save model if test loss has decreased
            if (valid_loss <= valid_loss_min) and (valid_loss != 0):
                print('Epoch: {} \tValidation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(epoch, valid_loss_min, valid_loss))
                torch.save(model.state_dict(), log_dir + '/model_bfs.pt')
                valid_loss_min = valid_loss

        scheduler.step()
    pass