import torch
import torch.nn as nn
import numpy as np


class AdmmLayer(nn.Module):
    """
    One layer of the unrolled ADMM network.
    """

    def __init__(self, im_height, im_width, im_channels, kernel_size, padding):
        super().__init__()

        # self.n_frames = n_frames
        self.im_height = im_height
        self.im_width = im_width

        self.p1 = nn.Conv2d(in_channels=im_channels,out_channels=im_channels,kernel_size=kernel_size,padding=padding, padding_mode='reflect')
        self.p2 = nn.Conv2d(in_channels=im_channels,out_channels=im_channels,kernel_size=kernel_size,padding=padding, padding_mode='reflect')
        self.p3 = nn.Conv2d(in_channels=im_channels,out_channels=im_channels,kernel_size=kernel_size,padding=padding, padding_mode='reflect')
        self.p4 = nn.Conv2d(in_channels=im_channels,out_channels=im_channels,kernel_size=kernel_size,padding=padding, padding_mode='reflect')


        lambda_from_pcp = 1/np.sqrt(im_height*im_width) # sqrt of len of vectorized image
        mu = .5  # assume mean abs value of input is .5

        self.lambda1 = nn.Parameter(torch.tensor([1/mu]))
        # self.lambda1 = nn.Parameter(torch.tensor([.5]))
        self.lambda2 = nn.Parameter(torch.tensor([.1*lambda_from_pcp/mu]))

        self.mu1 = nn.Parameter(torch.tensor([1/mu]))
        self.mu2 = nn.Parameter(torch.tensor([1/mu]))
        self.mu3 = nn.Parameter(torch.tensor([1/mu]))

        self.threshold = nn.Threshold(0,0)

    # parallel processing (k+1 is only dependent on k except lagrange multiplier)
    def forward(self, input):
        """
        :param input:
        :return:
        """

        (D, L, S, Y) = input

        # do S and L in parallel
        G1 = self.p2(D + Y/self.mu1 - self.p1(L))
        G2 = self.p4(D + Y/self.mu2 - self.p3(S))

        # create k+1 sparse component
        S_out = self.threshold(G1 - self.lambda1)

        # create k+1 low rank component
        G2 = torch.reshape(G2, (-1, self.im_height * self.im_width)).T
        (u, s, v) = torch.svd(G2)
        s = s - self.lambda2
        s = self.threshold(s)
        L_stacked = torch.mm(torch.mm(u, torch.diag(s)), v.t())
        L_out = L_stacked.T.reshape(-1, 1, self.im_height, self.im_width)

        # create k+1 lagrangian step
        Y_out = Y + self.mu3 * (D-self.p1(L_out)-self.p3(S_out))

        return D, L_out, S_out, Y_out


class AdmmNet(nn.Module):
    def __init__(self, shape, n_layers):
        super().__init__()
        self.n_channels = shape[1]
        self.im_height = shape[2]
        self.im_width = shape[3]

        self.layers = nn.ModuleList()
        for layer_idx in range(n_layers):
            if layer_idx < 3:
                kernel_size = 5
            else:
                kernel_size = 3
            padding = int(np.floor(kernel_size / 2))
            self.layers.append(AdmmLayer(self.im_height, self.im_width, self.n_channels, kernel_size, padding=padding))

    def forward(self, D):
        """
        :param input: tuple of D
        :return: tuple of L, S
        """
        S = torch.zeros_like(D)
        L = torch.zeros_like(D)
        Y = torch.zeros_like(D)

        components = (D, L, S, Y)

        for layer in self.layers:
            components = layer(components)
        (D, L, S, Y) = components

        return torch.cat((L, S), 1)
