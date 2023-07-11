"""
Implementation of PixelCNN

Reference: https://arxiv.org/pdf/1601.06759.pdf

Only grayscale input (MNIST) supported

"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskConv2d(nn.Conv2d):
    def __init__(self, mask='B',*args, **kargs):
        super(MaskConv2d, self).__init__(*args, **kargs)
        assert mask in {'A', 'B'}
        self.mask_type = mask
        self.register_buffer('mask', self.weight.data.clone())
        self.mask.fill_(1)
    
        _, _, H, W = self.mask.size()
    
        self.mask[:, :, H//2,W//2 + (self.mask_type == 'B'):] = 0
        self.mask[:, :, H//2+1:, :] = 0
    
    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskConv2d, self).forward(x)
    

class ResBlock(nn.Module):
    def __init__(self,h_dim):
        super(ResBlock,self).__init__()
        assert h_dim % 2 == 0
        self.h_dim = h_dim
        self.conv1 = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(h_dim,h_dim//2,1,1,0,bias=False),
            nn.BatchNorm2d(h_dim//2),
        )
        self.conv2 = nn.Sequential(
            nn.PReLU(),
            MaskConv2d('B',h_dim//2,h_dim//2,3,1,1,bias=False),
            nn.BatchNorm2d(h_dim//2),
        )
        self.conv3 = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(h_dim//2,h_dim,1,1,0,bias=False),
            nn.BatchNorm2d(h_dim),
        )

    def forward(self,x):
        identity = x
        res = self.conv1(x)
        res = self.conv2(res)
        res = self.conv3(res)
        outp = res + identity
        return outp

class PixelCNN(nn.Module):
    def __init__(self, num_layer,h_dim,binary_flag=True, *args, **kargs):
        super(PixelCNN, self).__init__(*args, **kargs)
        self.binary_flag = binary_flag
        self.layers = []
        for i in range(num_layer+1):
            if i == 0:
                self.layers.append(nn.Sequential(MaskConv2d('A',1,h_dim,7,1,3,bias=False),
                    nn.BatchNorm2d(h_dim)
                    ))
            else:
                self.layers.append(ResBlock(h_dim))
        self.layers.append(nn.Sequential(nn.PReLU(),
            nn.Conv2d(h_dim,h_dim,1,1,0,bias=False),
            nn.BatchNorm2d(h_dim),
            nn.PReLU(),
        ))
        self.backbone = nn.Sequential(*self.layers)
        if binary_flag:
            self.last_conv = nn.Conv2d(h_dim,1,1,1,0)
            self.prob = nn.Sigmoid()
        else:
           self.last_conv = nn.Conv2d(h_dim,256,1,1,0)
    def forward(self,x):
        x = self.backbone(x)
        x = self.last_conv(x)
        if self.binary_flag:
            x = self.prob(x)
        return x
