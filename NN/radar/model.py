"""
This model will serve as an RGB only baseline modeled after the architecture presented in:
Deep Unfolded Robust PCA with Application to Clutter Suppression in Ultrasound
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


class IstaLayer(nn.Module):
    """
    One layer of the unrolled ISTA network. See Fig. 1(b) in cited paper.
    Assuming images are grayscale
    """

    def __init__(self, im_height, im_width, im_channels, im_kernel_size, radar_kernel_size, im_padding, radar_padding):
        super().__init__()

        # self.n_frames = n_frames
        self.im_height = im_height
        self.im_width = im_width

        self.p1 = nn.Conv2d(in_channels=im_channels,out_channels=im_channels,kernel_size=im_kernel_size,padding=im_padding)
        self.p2 = nn.Conv2d(in_channels=im_channels,out_channels=im_channels,kernel_size=im_kernel_size,padding=im_padding)
        self.p3 = nn.Conv2d(in_channels=im_channels,out_channels=im_channels,kernel_size=im_kernel_size,padding=im_padding)
        self.p4 = nn.Conv2d(in_channels=im_channels,out_channels=im_channels,kernel_size=im_kernel_size,padding=im_padding)
        self.p5 = nn.Conv2d(in_channels=im_channels,out_channels=im_channels,kernel_size=im_kernel_size,padding=im_padding)
        self.p6 = nn.Conv2d(in_channels=im_channels,out_channels=im_channels,kernel_size=im_kernel_size,padding=im_padding)
        self.p7 = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=radar_kernel_size,padding=radar_padding)
        lambda_from_pcp = 1/np.sqrt(im_height*im_width) # sqrt of len of vectorized image
        mu = .5  # assume mean abs value of input is .5

        # notice in line below, mu is multiplied by 2
        self.lambda1 = nn.Parameter(torch.tensor([2/mu]))
        self.lambda2 = nn.Parameter(torch.tensor([.1*lambda_from_pcp/mu]))

        # self.lambda1 = nn.Parameter(torch.tensor([5.]))  # change
        # self.lambda2 = nn.Parameter(torch.tensor([.0003]))  # change

        self.threshold = nn.Threshold(0,0)

    def forward(self, input):
        """
        input is tuple of D,L,S
        :param D: Stacked images
        :param L: Stacked  low rank components
        :param S: Stacked sparse components
        :return: tuple of D, L^{k+1}, S^{k+1}

        notation for intermediate values will follow that of the cited paper.
        For example, L convolved with P_5 will be labeled L5
        """
        # print(self.lambda1, self.lambda2)
        (D,L,S,R) = input

        L5 = self.p5(L)
        L6 = self.p6(L)
        D1 = self.p1(D)
        D2 = self.p2(D)
        S3 = self.p3(S)
        S4 = self.p4(S)
        R7 = self.p7(R)

        R7_2dim = R7.unsqueeze(2).repeat(1, 1, self.im_height,1)

        L5_D1_S3 = L5+D1+S3
        L6_D2_S4 = L6+D2+S4
        L6_D2_S4_radar = L6_D2_S4 * R7_2dim

        L5_D1_S3 = torch.reshape(L5_D1_S3,(-1,self.im_height*self.im_width)).T

        (u,s,v) = torch.svd(L5_D1_S3)
        s = s - self.lambda1
        s = self.threshold(s)
        L_stacked = torch.mm(torch.mm(u, torch.diag(s)), v.t())

        D_out = D
        L_out = L_stacked.T.reshape(-1,1,self.im_height,self.im_width)
        S_out = torch.sign(L6_D2_S4_radar)*self.threshold(torch.abs(L6_D2_S4_radar)-self.lambda2)

        return D_out, L_out, S_out, R


class IstaNet(nn.Module):
        def __init__(self, shape, n_layers):
            super().__init__()
            self.n_channels = shape[1]
            self.im_height = shape[2]
            self.im_width = shape[3]

            self.layers = nn.ModuleList()
            for layer_idx in range(n_layers):
                if layer_idx < 3:
                    im_kernel_size = 5
                else:
                    im_kernel_size = 3
                im_padding = int(np.floor(im_kernel_size / 2))
                radar_kernel_size = 5
                radar_kernel_padding = int(np.floor(radar_kernel_size / 2))
                self.layers.append(IstaLayer(self.im_height, self.im_width, self.n_channels, im_kernel_size=im_kernel_size,
                                   im_padding=im_padding,  radar_kernel_size = radar_kernel_size,radar_padding=radar_kernel_padding))

        def forward(self, D, L, S, R):
            """
            :param input: tuple of D,L,S
            :return: tuple of D, L, S
            """
            components = (D,L,S,R)
            for layer in self.layers:
                components = layer(components)
            (D,L,S,R) = components

            return torch.cat((L,S),1)










