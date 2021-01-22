import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from model import IstaNet


def load_data(path):
    D = np.load(path + '/D.npy')
    L = np.load(path + '/L_pcp.npy')
    S = np.load(path + '/S_pcp.npy')

    # Add batch and channel dimension, new shape = [1,50,1,720,1280]
    D = D[None,:, None,::8,::8]
    L = L[None, :,None,::8,::8]
    S = S[None, :,None,::8,::8]

    return torch.from_numpy(D).float(), torch.from_numpy(L).float(), torch.from_numpy(S).float()

if __name__ == '__main__':
    # check if CUDA is available
    train_on_gpu = False # torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')

    # load Data (not dataset object since no train/test split)
    D,L,S = load_data('/home/spencer/research/radar-rgb-bfs/output')
    target = torch.cat((L,S),2)

    # init model
    # (batch_size, n_frames,n_channels,im_height,im_width) = D.shape
    (batch_size, n_frames,n_channels,im_height,im_width) = D.shape
    model = IstaNet(n_frames,im_height,im_width)
    if train_on_gpu:
        model.cuda()
    # specify loss function (categorical cross-entropy)
    criterion = nn.MSELoss()

    # specify optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # init L, S
    L = torch.zeros_like(D)
    S = torch.zeros_like(D)

    # train

    n_epochs = 70  # number of epochs to train the model
    n_batch = 1  # number of batches. this will be changed but using it for training loss avg
    valid_loss_min = np.Inf  # track change in validation loss

    for epoch in range(1, n_epochs + 1):

        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        model.train()
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            D, L, S, target = D.cuda(), L.cuda(), L.cuda(), target.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # test on multiple sequences per batch i.e. 5 batches each with 50 frames
        for seq in range(D.shape[0]):
            D_seq = D[seq]
            L_seq = L[seq]
            S_seq = S[seq]
            target_seq = target[seq]

            # forward pass: compute predicted outputs by passing inputs to the model
            output = model((D_seq,L_seq,S_seq))
            # calculate the batch loss
            loss = criterion(output, target_seq)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item() * D_seq.size(0)

        # ######################
        # # validate the model #
        # ######################
        # model.eval()
        # for data, target in valid_loader:
        #     # move tensors to GPU if CUDA is available
        #     if train_on_gpu:
        #         data, target = data.cuda(), target.cuda()
        #     # forward pass: compute predicted outputs by passing inputs to the model
        #     output = model(data)
        #     # calculate the batch loss
        #     loss = criterion(output, target)
        #     # update average validation loss
        #     valid_loss += loss.item() * data.size(0)

        # calculate average losses
        train_loss = train_loss / (batch_size * n_frames * n_batch)   # len(train_loader.sampler)
        # valid_loss = valid_loss / len(valid_loader.sampler)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(
            epoch, train_loss))

        # save model if validation loss has decreased
        # if valid_loss <= valid_loss_min:
        #     print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        #         valid_loss_min,
        #         valid_loss))
        #     torch.save(model.state_dict(), 'model_cifar.pt')
        #     valid_loss_min = valid_loss
        if epoch % 10 ==0:
            plt.figure()
            plt.subplot(221)
            plt.imshow(target_seq.numpy()[19, 0])
            plt.title('target L')
            plt.subplot(222)
            plt.imshow(output.detach().numpy()[19,0])
            plt.title('L')
            plt.subplot(223)
            plt.title('target S')
            plt.imshow(target_seq.numpy()[19, 1])
            plt.subplot(224)
            plt.imshow(output.detach().numpy()[19,1])
            plt.title('S')
            plt.suptitle(epoch)
            plt.show()