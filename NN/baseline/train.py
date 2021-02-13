import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import IstaNet
from torch.utils.tensorboard import SummaryWriter
import itertools as it
from tensorboard_helper import plot_classes_preds
from skimage.transform import rescale
import argparse
import yaml
from datetime import datetime
import shutil

import matplotlib.pyplot as plt


def load_data(path, n_frames = 30, rescale_factor = 1.):

    D = np.load(path + '/D.npy')
    L = np.load(path + '/L_pcp.npy')
    S = np.load(path + '/S_pcp.npy')

    # Add channel dimension, new shape = [n_frames,1,720,1280] not including downsampling
    D = D[:n_frames, None,:,:]
    L = L[:n_frames, None,:,:]
    S = S[:n_frames, None,:,:]

    # Downsample with anti-aliasing filter
    if rescale_factor != 1:
        D = rescale(D,(1,1,rescale_factor,rescale_factor),anti_aliasing=True)
        L = rescale(L,(1,1,rescale_factor,rescale_factor),anti_aliasing=True)
        S = rescale(S,(1,1,rescale_factor,rescale_factor),anti_aliasing=True)

    return torch.from_numpy(D).float(), torch.from_numpy(L).float(), torch.from_numpy(S).float()


def random_crop(D, target, patch_shape):
    (full_height, full_width) = D.shape[-2:]
    (patch_height, patch_width) = patch_shape
    height_range = (patch_height//2,full_height-patch_height//2)
    width_range = (patch_width//2,full_width-patch_width//2)

    rand_height = np.random.choice(np.arange(height_range[0],height_range[1]))
    rand_width = np.random.choice(np.arange(width_range[0],width_range[1]))

    D_patch = D[:,:,rand_height-patch_height//2:rand_height+patch_height//2,rand_width-patch_width//2:rand_width+patch_width//2]
    target_patch = target[:,:,rand_height-patch_height//2:rand_height+patch_height//2,rand_width-patch_width//2:rand_width+patch_width//2]

    return D_patch, target_patch


def infer_full_image(full_input, network, patch_shape, device):
    (height_step, width_step) = np.array(patch_shape[-2:])
    (patch_height,patch_width) = np.array(patch_shape[-2:])
    (_,_,og_height, og_width) = full_input.shape
    # full_output = [ [] for _ in range(og_height*og_width)]
    full_output = torch.zeros(full_input.shape[0],2,og_height,og_width)
    for y in range(patch_height//2,og_height+height_step,height_step):
        y = np.min([y,og_height-patch_height//2])
        for x in range(patch_width//2,og_width+width_step,width_step):
            x = np.min([x,og_width-patch_width//2])
            print(y,x)
            # twoD_indexes = list(it.product(np.arange(y-patch_height//2,y+patch_height//2),
            #                                 np.arange(x-patch_width//2,x+patch_width//2)))
            # oneD_indexes = [ii[0]*og_width+ii[1] for ii in twoD_indexes]
            # patch_input = full_input[:,:,y-patch_height//2:y+patch_height//2,x-patch_width//2:x+patch_width//2].to(device)
            # L_patch_input = torch.zeros_like(patch_input).to(device)
            # S_patch_input = torch.zeros_like(patch_input).to(device)
            # patch_out = network(patch_input, L_patch_input, S_patch_input)
            # patch_out = patch_out.reshape(patch_out.shape[0],patch_out.shape[1],-1).detach().cpu().numpy()
            #
            # for idx, oneD_index in enumerate(oneD_indexes):
            #     full_output[oneD_index].append(patch_out[:,:,idx])

            patch_input = full_input[:, :, y - patch_height // 2:y + patch_height // 2, x - patch_width // 2:x + patch_width // 2].to(device)
            L_patch_input = torch.zeros_like(patch_input).to(device)
            S_patch_input = torch.zeros_like(patch_input).to(device)
            patch_out = network(patch_input, L_patch_input, S_patch_input).detach().cpu()
            full_output[:,:, y - patch_height // 2:y + patch_height // 2, x - patch_width // 2:x + patch_width // 2] = patch_out

    return full_output


if __name__ == '__main__':
    torch.manual_seed(17761948)

    parser = argparse.ArgumentParser()
    parser.add_argument("-yaml", help="path of yaml file", type=str)
    args = parser.parse_args()
    yampl_path = args.yaml

    with open(yampl_path) as file:
        setup_dict = yaml.load(file,Loader=yaml.FullLoader)
    train_path = setup_dict['train_path'][0]
    test_path = setup_dict['test_path'][0]
    n_layers = setup_dict['n_layers'][0]
    try_gpu = setup_dict['try_gpu'][0]
    downsample_rate = setup_dict['downsample'][0]
    learning_rate = setup_dict['lr'][0]
    schedule_step = setup_dict['schedule_step'][0]
    schedule_multiplier = setup_dict['schedule_multiplier'][0]
    patch_height = setup_dict['patch_height'][0]
    patch_width = setup_dict['patch_width'][0]

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
    D_train_full,L_train_full_target,S_train_full_target = load_data(train_path,rescale_factor=downsample_rate)
    target_train_full = torch.cat((L_train_full_target,S_train_full_target),1)  # concatenate the L and S in the channel dimension
    D_train_full.to(device)

    D_test_full, L_test_full_target, S_test_full_target = load_data(test_path, rescale_factor=downsample_rate)
    target_test_full = torch.cat((L_test_full_target, S_test_full_target), 1)  # concatenate the L and S in the channel dimension
    D_test_full.to(device)

    # Destination for tensorboard log data
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    log_name = 'OG_l'+str(n_layers) + '_lr_' + str(learning_rate) + '_ds_' + str(downsample_rate)
    log_dir = '../runs/'+log_name+'__'+dt_string
    writer = SummaryWriter(log_dir)
    shutil.copyfile(yampl_path, log_dir + '/setup.yaml')


    # init model
    data_shape = list(np.concatenate([D_train_full.shape[:2],[patch_height,patch_width]]))
    model = IstaNet(data_shape,n_layers)
    model.to(device)
    # specify loss function (categorical cross-entropy)
    criterion = nn.MSELoss()
    # specify optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer,gamma=schedule_multiplier,step_size=schedule_step)

    # tensorboard graph
    writer.add_graph(model,(D_train_full[:,:,:patch_height,:patch_width].to(device),D_train_full[:,:,:patch_height,:patch_width].to(device),
                            D_train_full[:,:,:patch_height,:patch_width].to(device)))

    writer.close()

    # train

    n_epochs = 10000  # number of epochs to train the model
    valid_loss_min = np.Inf  # track change in validation loss

    for epoch in range(1, n_epochs + 1):

        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        model.train()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()

        # init L, S as zeros per PCP algorithm
        L_train_patch = torch.zeros(data_shape)
        S_train_patch = torch.zeros(data_shape)

        # get random crop
        D_train_patch, target_train_patch = random_crop(D_train_full,target_train_full,(patch_height,patch_width))

        D_train_patch, L_train_patch, S_train_patch, target_train_patch = D_train_patch.to(device), L_train_patch.to(device), S_train_patch.to(device), target_train_patch.to(device)

        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(D_train_patch,L_train_patch,S_train_patch)
        # calculate the batch loss
        loss = criterion(output, target_train_patch)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()

        # add training loss in tensorboard
        writer.add_scalar('training pixel loss',
                          loss.item(),
                          epoch)
        writer.close()

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

        # # add lambda1 (SVD lambda)
        # writer.add_scalar('Lambda1 (SVD) in last layer', model.layers[-1].lambda1.item(), epoch)
        # writer.close()
        #
        # # add lambda2 (Shrink S)
        # writer.add_scalar('Lambda2 Shrink S in last layer', model.layers[-1].lambda2.item(), epoch)
        # writer.close()

        # show sample predictions
        if epoch % 250 ==0:
            # memory saver
            del L_train_patch, S_train_patch #L_flat, u, s, v

            model.eval()
            output_train_full = infer_full_image(D_train_full,model,data_shape,device)
            writer.add_figure('predictions vs. actuals TRAIN',
                          plot_classes_preds(output_train_full.cpu().detach().numpy(),L_train_full_target.cpu().numpy(),S_train_full_target.cpu().numpy()),
                          global_step=epoch)
            writer.close()



            ######################
            # validate the model #
            ######################

            # L_test_patch = torch.zeros(data_shape)
            # S_test_patch = torch.zeros(data_shape)
            #
            # # move tensors to GPU if CUDA is available
            # D_test, L_test, S_test, target_test = D_test.to(device), L_test.to(device), S_test.to(device), target_test.to(device)

            # # forward pass: compute predicted outputs by passing inputs to the model
            # output_test = model(D_test,L_test,S_test)
            # # calculate the batch loss
            # loss = criterion(output_test, target_test)
            # # update average validation loss
            # valid_loss += loss.item()

            output_test_full = infer_full_image(D_test_full, model, data_shape, device)
            writer.add_figure('predictions vs. actuals TEST',
                              plot_classes_preds(output_test_full.cpu().detach().numpy(), L_test_full_target.cpu().numpy(),
                                                 S_test_full_target.cpu().numpy()),
                              global_step=epoch)
            writer.close()

            # writer.add_figure('predictions vs. actuals TEST',
            #                   plot_classes_preds(output_test.cpu().detach().numpy(), L_test_target.cpu().numpy(), S_test_target.numpy()),
            #                   global_step=epoch)
            # writer.close()
            # print('Epoch: {} \tTraining Loss: {:.6f} \tTest Loss: {:.6f}'.format(
            #     epoch, train_loss, valid_loss))
            #
            # # save model if validation loss has decreased
            # if (valid_loss <= valid_loss_min) and (valid_loss != 0):
            #     print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            #         valid_loss_min,
            #         valid_loss))
            #     torch.save(model.state_dict(), log_dir + '/model_bfs.pt')
            #     valid_loss_min = valid_loss
            #
            # # add test loss in tensorboard
            # writer.add_scalar('test pixel loss',
            #                   loss.item(),
            #                   epoch)
            # writer.close()
            #
            # del output_test, loss, L_test, S_test
        else:
            print('Epoch: {} \tTraining Loss: {:.6f}'.format(
                epoch, train_loss))


        scheduler.step()
