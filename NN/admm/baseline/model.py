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
        self.lambda2 = nn.Parameter(torch.tensor([.1*lambda_from_pcp/mu]))

        self.mu1 = nn.Parameter(torch.tensor([1/mu]))
        self.mu2 = nn.Parameter(torch.tensor([1/mu]))
        self.mu3 = nn.Parameter(torch.tensor([1/mu]))

        self.threshold = nn.Threshold(0,0)

    # parallel processing (k+1 is only dependent on k)
    def forward(self, input):
        """
        :param input:
        :return:
        """

        (D, L, S, Y) = input

        L1 = self.p1(L)
        L2 = self.p2(L)

        Y1 = self.mu1 * Y
        Y2 = self.mu2 * Y

        S3 = self.p3 * S
        S4 = self.p4 * S

        # create k+1 low rank component
        D_S4_Y2 = D - S4 - Y2
        D_S4_Y2 = torch.reshape(D_S4_Y2, (-1, self.im_height * self.im_width)).T
        (u, s, v) = torch.svd(D_S4_Y2)
        s = s - self.lambda1
        s = self.threshold(s)
        L_stacked = torch.mm(torch.mm(u, torch.diag(s)), v.t())
        L_out = L_stacked.T.reshape(-1, 1, self.im_height, self.im_width)

        # create k+1 sparse component
        D_L1_Y1 = D - L1 + Y1
        S_out = self.threshold(D_L1_Y1-self.lambda2)

        # create k+1 lagrangian step
        Y_out = Y + self.mu3 * (D-L2-S3)

        return (D, L_out, S_out, Y_out)

    # sequential processing (k+1 is dependent on other k+1 terms)
    # TODO: If we keep this, then we can get rid of some conv layers
    def forward(self, input):
        """
        :param input:
        :return:
        """

        (D, L, S, Y) = input

        L1 = self.p1(L)
        L2 = self.p2(L)

        Y1 = self.mu1 * Y
        Y2 = self.mu2 * Y

        S3 = self.p3 * S
        S4 = self.p4 * S


        # create k+1 low rank component
        D_S4_Y2 = D - S4 - Y2
        D_S4_Y2 = torch.reshape(D_S4_Y2,(-1,self.im_height*self.im_width)).T

        (u, s, v) = torch.svd(D_S4_Y2)
        s = s - self.lambda1
        s = self.threshold(s)
        L_stacked = torch.mm(torch.mm(u, torch.diag(s)), v.t())
        L_out = L_stacked.T.reshape(-1, 1, self.im_height, self.im_width)

        # create k+1 sparse component
        D_L1_Y1 = D - L_out + Y1
        S_out = self.threshold(D_L1_Y1 - self.lambda2)

        # create k+1 lagrangian step
        Y_out = Y + self.mu3 * (D-L_out-S_out)

        return D, L_out, S_out, Y_out

    class ADMMNet(nn.Module):
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