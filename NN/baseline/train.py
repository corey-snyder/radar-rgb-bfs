import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from model import IstaNet
from torch.utils.tensorboard import SummaryWriter
from tensorboard_helper import plot_classes_preds


def load_data(path):
    D = np.load(path + '/D.npy')
    L = np.load(path + '/L_pcp.npy')
    S = np.load(path + '/S_pcp.npy')

    # Add channel dimension, new shape = [50,1,720,1280] not including downsampling
    D = D[:30, None,::8,::8]
    L = L[:30,None,::8,::8]
    S = S[:30,None,::8,::8]

    return torch.from_numpy(D).float(), torch.from_numpy(L).float(), torch.from_numpy(S).float()

if __name__ == '__main__':
    # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')

    # load Data (not dataset object since no train/test split)
    D_train,L_train_target,S_train_target = load_data('/home/spencer/research/radar-rgb-bfs/output/csl_lobby_700')
    target_train = torch.cat((L_train_target,S_train_target),1)  # concatenate the L and S in the channel dimension

    D_test, L_test_target, S_test_target = load_data('/home/spencer/research/radar-rgb-bfs/output/csl_lobby_side_0')
    target_test = torch.cat((L_test_target, S_test_target), 1)  # concatenate the L and S in the channel dimension

    # Destination for tensorboard log data
    writer = SummaryWriter('runs/blah')

    # init model
    (n_frames,n_channels,im_height,im_width) = D_train.shape
    model = IstaNet(im_height,im_width)
    if train_on_gpu:
        model.cuda()
    # specify loss function (categorical cross-entropy)
    criterion = nn.MSELoss()

    # specify optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # init L, S as zeros per PCP algorithm
    L_train = torch.zeros_like(D_train)
    S_train = torch.zeros_like(D_train)
    L_test = torch.zeros_like(D_test)
    S_test = torch.zeros_like(D_test)

    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        D_train, L_train, S_train, target_train = D_train.cuda(), L_train.cuda(), S_train.cuda(), target_train.cuda()
        D_test, L_test, S_test, target_test = D_test.cuda(), L_test.cuda(), S_test.cuda(), target_test.cuda()

    # tensorboard graph
    writer.add_graph(model,(D_train,L_train,S_train))
    writer.close()

    # train

    n_epochs = 1000  # number of epochs to train the model
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
        # test on multiple sequences per batch i.e. 5 batches each with 50 frames

        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(D_train,L_train,S_train)
        # calculate the batch loss
        loss = criterion(output, target_train)
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



        # add values of singular values of L in tensorboard
        L_flat = torch.reshape(output[:,0], (-1, im_height * im_width)).T
        (u, s, v) = torch.svd(L_flat)
        s_dict = {}
        for idx, ii in enumerate(s):
            s_dict[str(idx)] = s[idx].item()
        writer.add_scalars('sing. vals of L', s_dict,epoch)
        writer.close()

        # add rank of L to tensorboard
        writer.add_scalar('rank of L', sum(s>1e-4),epoch)

        # add lambda1 (SVD lambda)
        writer.add_scalar('Lambda1 (SVD) in last layer', model.ista8.lambda1.item(), epoch)
        writer.close()

        # add lambda2 (Shrink S)
        writer.add_scalar('Lambda2 Shrink S in last layer', model.ista8.lambda2.item(), epoch)
        writer.close()

        # show sample predictions
        if epoch % 20:
            # if train_on_gpu:
                # output = output.cpu(), L_train_target.cpu()
            writer.add_figure('predictions vs. actuals TRAIN',
                          plot_classes_preds(output.cpu().detach().numpy(),L_train_target.cpu().numpy(),S_train_target.cpu().numpy()),
                          global_step=epoch)
            writer.close()

        ######################
        # validate the model #
        ######################
        if epoch % 20 ==0:
            model.eval()
            # forward pass: compute predicted outputs by passing inputs to the model
            output_test = model(D_test,L_test,S_test)
            # calculate the batch loss
            loss_test = criterion(output, target_test)
            # update average validation loss
            valid_loss += loss_test.item()

            writer.add_figure('predictions vs. actuals TEST',
                              plot_classes_preds(output_test.cpu().detach().numpy(), L_test_target.cpu().numpy(), S_test_target.numpy()),
                              global_step=epoch)
            del output_test,loss_test
            writer.close()
            print('Epoch: {} \tTraining Loss: {:.6f} \tTest Loss: {:.6f}'.format(
                epoch, train_loss, valid_loss))
        else:
            print('Epoch: {} \tTraining Loss: {:.6f}'.format(
                epoch, train_loss))
        # calculate average losses
        # train_loss = train_loss  # average loss per pixel
        # valid_loss = valid_loss / len(valid_loader.sampler)

        # save model if validation loss has decreased
        if (valid_loss <= valid_loss_min) and (valid_loss != 0):
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            torch.save(model.state_dict(), 'model_bfs.pt')
            valid_loss_min = valid_loss