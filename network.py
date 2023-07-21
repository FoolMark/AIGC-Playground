"""
Implementation of Gated PixelCNN

Reference: https://arxiv.org/pdf/1606.05328.pdf

Only grayscale input (MNIST) supported

Inspired by the follow github repo
https://github.com/anordertoreclaim/PixelCNN/tree/master

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

#Vertical Conv
class VConv(nn.Conv2d):
    def __init__(self,*args, **kargs):
        super(VConv, self).__init__(*args, **kargs)
        self.H,_ = self.kernel_size

    def forward(self, x):
        x = super(VConv, self).forward(x)

        v = x[:,:,1:-self.H,:]
        shift_v = x[:,:,:-self.H-1,:]

        return v,shift_v

#Horizontal Conv
class HConv(nn.Conv2d):
    def __init__(self,*args,mask, **kargs):
        super(HConv, self).__init__(*args, **kargs)
        assert mask in {'A', 'B'}

        self.mask_type = mask
        self.register_buffer('mask', self.weight.data.clone())
        self.mask.fill_(1)

        _, _,_, W = self.mask.size()
        self.mask[:, :,:, W//2 + (self.mask_type == 'B'):] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(HConv, self).forward(x)


class CasualBlock(nn.Module):
    def __init__(self,in_dim,h_dim):
        super(CasualBlock,self).__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        k_size = 7

        self.v_conv = VConv(in_dim,2*h_dim,(k_size//2+1,k_size),1,(k_size//2+1,k_size//2))
        self.h_conv = HConv(in_dim,2*h_dim,(1,k_size),1,(0,k_size//2),mask='A')

        self.v2h = nn.Conv2d(2*h_dim,2*h_dim,1)
        self.h_fc = nn.Conv2d(h_dim,h_dim,1)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        #vertical
        v_out,v_out_shift = self.v_conv(x)
        v_out1,v_out2 = torch.split(v_out,self.h_dim,dim=1)
        v_out = self.tanh(v_out1)*self.sigmoid(v_out2)

        #horizontal
        h_out = self.h_conv(x)
        v_out_shift = self.v2h(v_out_shift)
        h_out = h_out + v_out_shift
        h_out1,h_out2 = torch.split(h_out,self.h_dim,dim=1)
        h_out = self.tanh(h_out1)*self.sigmoid(h_out2)

        h_out = self.h_fc(h_out)

        return v_out,h_out


class GatedBlock(nn.Module):
    def __init__(self,in_dim,h_dim,k_size=5):
        super(GatedBlock,self).__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.k_size = k_size

        self.v_conv = VConv(in_dim,2*h_dim,(k_size//2+1,k_size),1,(k_size//2+1,k_size//2))
        self.h_conv = HConv(in_dim,2*h_dim,(1,k_size),1,(0,k_size//2),1,mask='B')
        self.h_skip = nn.Conv2d(h_dim,h_dim,1) 
        
        self.v2h = nn.Conv2d(2*h_dim,2*h_dim,1)
        self.h_fc = nn.Conv2d(h_dim,h_dim,1)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        v_in,h_in,skip = x[0],x[1],x[2]
        
        #vertical
        v_out,v_out_shift = self.v_conv(v_in)
        v_out1,v_out2 = torch.split(v_out,self.h_dim,dim=1)
        v_out = self.tanh(v_out1)*self.sigmoid(v_out2)

        #horizontal
        h_out = self.h_conv(h_in)
        v_out_shift = self.v2h(v_out_shift)
        h_out = h_out + v_out_shift
        h_out1,h_out2 = torch.split(h_out,self.h_dim,dim=1)
        h_out = self.tanh(h_out1)*self.sigmoid(h_out2)

        skip = skip + self.h_skip(h_out)

        h_out = self.h_fc(h_out)
        h_out = h_out + h_in

        return [v_out,h_out,skip]

class PixelCNN(nn.Module):
    def __init__(self,num_layer,h_dim,k_size,color_bit,*args, **kargs):
        super(PixelCNN, self).__init__(*args, **kargs)
        self.casual = CasualBlock(1,h_dim)
        self.layers = []
        for i in range(num_layer):
            self.layers.append(GatedBlock(h_dim,h_dim,k_size))
        self.gateConv = nn.Sequential(*self.layers)
        self.outConv = nn.Sequential(nn.PReLU(),
            nn.Conv2d(h_dim,h_dim,1,1,0,bias=False),
            nn.BatchNorm2d(h_dim),
            nn.PReLU())
        
        self.lastConv = nn.Conv2d(h_dim,2**color_bit,1,1,0)

    def forward(self,x):
        v,h = self.casual(x)
        blank = torch.zeros(x.shape,requires_grad=True).cuda()
        x = [v,h,blank]
        _,_,skip = self.gateConv(x)
        x = self.outConv(skip)
        x = self.lastConv(x)
        return x

