from utils import *
from config import *
from network import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import pdb
import matplotlib.pyplot as plt

def display(x,TX,TY):
    x = x.detach().cpu().numpy()
    h1,h2,w1,w2 = -1,-1,-1,-1
    for i in range(28):
        for j in range(28):
            if abs(x[i][j]) != 0:
                if h1 < 0:
                    h1 = i
                h2 = max(h2,i)
                if w1 < 0:
                    w1 = j
                w2 = max(w2,j)
    #range of receptive field
    print(h1,h2,w1,w2)
    
    #center
    print(abs(x[TX][TY])>0,x[14][14])
    plt.imshow(x,cmap='binary',interpolation='nearest')
    plt.plot([w1-0.5,w2+0.5],[h1-0.5,h1-0.5],color='red')
    plt.plot([w1-0.5,w1-0.5],[h1-0.5,h2+0.5],color='red')
    plt.plot([w2+0.5,w2+0.5],[h1-0.5,TX-0.5],color='red')
    plt.plot([w1-0.5,TY-0.5],[TX+0.5,TX+0.5],color='red')
    plt.plot([TY-0.5,TY-0.5],[TX+0.5,TX-0.5],color='red')
    plt.plot([TY-0.5,w2+0.5],[TX-0.5,TX-0.5],color='red')



    plt.colorbar()
    plt.savefig('grad.png')

def generate_conv(*args,**kargs):
    mod = nn.Conv2d(*args,**kargs)
    mod.weight.data.fill_(1.0)
    return mod

def generate_casual_block(h_dim):
    mod = CasualBlock(1,h_dim)
    mod.v_conv.weight.data.fill_(1.0)
    mod.h_conv.weight.data.fill_(1.0)
    return mod

def generate_gate_block(h_dim):
    mod = GatedBlock(h_dim,h_dim,k_size)
    mod.v_conv.weight.data.fill_(1.0)
    mod.h_conv.weight.data.fill_(1.0)
    return mod

def generate_pixelcnn(h_dim,n_layer):
    cas = generate_casual_block(h_dim)
    mods = []
    for i in range(n_layer):
        mods.append(generate_gate_block(h_dim))
    return cas,nn.Sequential(*mods)   

INPUT_SIZE = 28
N_LAYER = 5

mods = []
cas,gate = generate_pixelcnn(1,N_LAYER)
INPUT = torch.rand(10,1,INPUT_SIZE,INPUT_SIZE).requires_grad_()
V,H = cas(INPUT)
B = torch.zeros(INPUT.shape,requires_grad=True)
X = [V,H,B]
_,_,X = gate(X)
print(X.shape)

TX,TY = 14,14

loss = torch.sum(X[:,:,TX,TY] - 1000.0)**5
loss.backward()

tmp = INPUT.grad[0][0]
display(tmp,TX,TY)