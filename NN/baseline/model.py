"""
This model will serve as an RGB only baseline modeled after the architecture presented in:
Deep Unfolded Robust PCA with Application to Clutter Suppression in Ultrasound
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IstaLayer(nn.Module):
    """
    One layer of the unrolled ISTA network. See Fig. 1(b) in cited paper.
    Assuming images are grayscale
    """

    def __init__(self, n_frames, im_height, im_width, kernel_size, padding):
        super().__init__()

        self.n_frames = n_frames
        self.im_height = im_height
        self.im_width = im_width

        self.p1 = nn.Conv2d(in_channels=n_frames,out_channels=n_frames,kernel_size=kernel_size,padding=padding)
        self.p2 = nn.Conv2d(in_channels=n_frames,out_channels=n_frames,kernel_size=kernel_size,padding=padding)
        self.p3 = nn.Conv2d(in_channels=n_frames,out_channels=n_frames,kernel_size=kernel_size,padding=padding)
        self.p4 = nn.Conv2d(in_channels=n_frames,out_channels=n_frames,kernel_size=kernel_size,padding=padding)
        self.p5 = nn.Conv2d(in_channels=n_frames,out_channels=n_frames,kernel_size=kernel_size,padding=padding)
        self.p6 = nn.Conv2d(in_channels=n_frames,out_channels=n_frames,kernel_size=kernel_size,padding=padding)

        self.lambda1 = nn.Parameter(torch.tensor([.1]))  # change
        self.lambda2 = nn.Parameter(torch.tensor([.01]))  # change

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
        print(self.lambda1, self.lambda2)
        (D,L,S) = input

        L5 = self.p5(L)
        L6 = self.p6(L)
        D1 = self.p1(D)
        D2 = self.p2(D)
        S3 = self.p3(S)
        S4 = self.p4(S)

        L5_D1_S3 = L5+D1+S3
        L6_D2_S4 = L6+D2+S4

        L5_D1_S3 = torch.reshape(L5_D1_S3,(-1,self.im_height*self.im_width)).T
        # L5_D1_S3 = L5_D1_S3.permute(0,2,1)

        (u,s,v) = torch.svd(L5_D1_S3)
        s = s - self.lambda1
        s = self.threshold(s)
        L_stacked = torch.mm(torch.mm(u, torch.diag(s)), v.t())

        D_out = D
        L_out = L_stacked.T.reshape(-1,self.n_frames,self.im_height,self.im_width)
        S_out = torch.sign(L6_D2_S4)*self.threshold(torch.abs(L6_D2_S4)-self.lambda2)

        return D_out, L_out, S_out


class IstaNet(nn.Module):
        def __init__(self, n_frames, im_height, im_width):
            super().__init__()
            self.n_frames = 1# n_frames
            self.im_height = im_height
            self.im_width = im_width

            self.ista1 = IstaLayer(self.n_frames, im_height, im_width, 5,padding=(2,2))
            self.ista2 = IstaLayer(self.n_frames, im_height, im_width, 5,padding=(2,2))
            self.ista3 = IstaLayer(self.n_frames, im_height, im_width, 5,padding=(2,2))
            self.ista4 = IstaLayer(self.n_frames, im_height, im_width, 3,padding=(1,1))
            self.ista5 = IstaLayer(self.n_frames, im_height, im_width, 3,padding=(1,1))
            self.ista6 = IstaLayer(self.n_frames, im_height, im_width, 3,padding=(1,1))
            self.ista7 = IstaLayer(self.n_frames, im_height, im_width, 3,padding=(1,1))
            self.ista8 = IstaLayer(self.n_frames, im_height, im_width, 3,padding=(1,1))
            self.ista9 = IstaLayer(self.n_frames, im_height, im_width, 3,padding=(1,1))
            self.ista10 = IstaLayer(self.n_frames, im_height, im_width, 3,padding=(1,1))

        def forward(self,input):
            """
            :param input: tuple of D,L,S
            :return: tuple of D, L, S
            """
            components = self.ista1(input)
            components = self.ista2(components)
            components = self.ista3(components)
            components = self.ista4(components)
            components = self.ista5(components)
            components = self.ista6(components)
            # components = self.ista7(components)
            # components = self.ista8(components)
            # components = self.ista9(components)
            # components = self.ista10(components)

            (D,L,S) = components

            return torch.cat((L,S),1)











