import numpy as np
import torch
from skimage.transform import rescale


def load_rgb_data(path, n_frames, rescale_factor = 1., iterations=400, print_flag=True):

    D = np.load(path + '/D.npy')
    L = np.load(path + '/L_ista_{}_{}.npy'.format(iterations,n_frames))
    S = np.load(path + '/S_ista_{}_{}.npy'.format(iterations,n_frames))

    if print_flag:
        print('\n'+path + '/D.npy')
        print(path + '/L_ista_{}_{}.npy'.format(iterations,n_frames))
        print(path + '/S_ista_{}_{}.npy'.format(iterations, n_frames))

    # Add channel dimension, new shape = [n_frames,1,720,1280] not including downsampling
    D = D[:n_frames, None,:,:]
    L = L[:, None,:,:]
    S = S[:, None,:,:]

    # Downsample with anti-aliasing filter
    if rescale_factor != 1:
        D = rescale(D,(1,1,rescale_factor,rescale_factor),anti_aliasing=True)
        L = rescale(L,(1,1,rescale_factor,rescale_factor),anti_aliasing=True)
        S = rescale(S,(1,1,rescale_factor,rescale_factor),anti_aliasing=True)

    return torch.from_numpy(D).float(), torch.from_numpy(L).float(), torch.from_numpy(S).float()


def load_radar_rgb_data(path, n_frames, rescale_factor = 1., iterations = 400, print_flag = True):

    D, L, S = load_rgb_data(path, n_frames, rescale_factor, iterations, print_flag)

    R = np.load(path + '/radar_likelihoods.npy')

    if print_flag:
        print(path + '/radar_likelihoods.npy')

    # Add channel dimension, new shape = [n_frames,1,1280] not including downsampling
    R = R[:n_frames, None,:]

    # Downsample with anti-aliasing filter
    if rescale_factor != 1:
        R = rescale(R,(1,1,rescale_factor),anti_aliasing=True)

    return D, L, S, torch.from_numpy(R).float()


def threshold_data(input, thresh):
    input_abs = torch.abs(input)
    input_abs[input_abs < thresh] = 0
    input_abs[input_abs >= thresh] = 1

    return input_abs


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    path = '../../Data/CSL_lobby_side_0/raw'
    idx = 45
    D, L, S = load_rgb_data(path,50,rescale_factor=.25)
    D, L, S = D.detach().numpy(), L.detach().numpy(), S.detach().numpy()
    plt.figure()
    plt.subplot(131)
    plt.imshow(D[idx,0])
    plt.subplot(132)
    plt.imshow(L[idx, 0])
    plt.subplot(133)
    plt.imshow(S[idx, 0])

    print()

    D, L, S, R = load_radar_rgb_data(path,50,rescale_factor=.25)
    D, L, S, R = D.detach().numpy(), L.detach().numpy(), S.detach().numpy(), R.detach().numpy()
    plt.figure()
    plt.subplot(141)
    plt.imshow(D[idx,0])
    plt.subplot(142)
    plt.imshow(L[idx, 0])
    plt.subplot(143)
    plt.imshow(S[idx, 0])
    plt.subplot(144)
    plt.plot(R[idx, 0])

    plt.show()
