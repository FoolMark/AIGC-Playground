import os
import torch
import logging
import torchvision
from torchvision import datasets,transforms
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
import numpy as np

transformGrey = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.,), (1.0,))
])

transformColor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.,0.,0.), (1.0,1.0,1.0))
])

def trainLoader(dset='MNIST',bs=100):
    if dset == 'MNIST':
        train_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transformGrey)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
    return train_loader

def testLoader(dset='MNIST',bs=100):
    if dset == 'MNIST':
        test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=transformGrey)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)
    return test_loader

def setLogger():
    logging.basicConfig(level=logging.INFO, filename='log/worklog.txt',\
     filemode='w', format='%(asctime)s [%(levelname)s] %(message)s')
    logger = logging.getLogger('train_log')
    return logger

def setTB():
    tbWriter = SummaryWriter('log/tb')
    return tbWriter

def initial():
    os.makedirs('log',exist_ok=True)
    os.makedirs('sample',exist_ok=True)
    os.makedirs('checkpoint',exist_ok=True)
    logger = setLogger()
    tbWriter = setTB()
    return logger,tbWriter

def adjust_lr(opt,lr,epoch,max_epoch):
    factor =  (1 + np.cos((epoch+1)/max_epoch*2*np.pi)) / 2
    for param_group in opt.param_groups:
        param_group['lr'] = lr * factor
    return lr * factor

def binarize(x,sample=False):
    if sample == True:
        y = torch.rand(x.shape).cuda()
        return (y<x).float()
    else:
        y = (x >= 0.5)
        return y.float()

def quantize(x,bit=8,sample=False):
    M = 2**bit - 1
    if sample == False:
        y = (x * M).to(torch.uint8)
        return y
    else:
        x = x.permute(0,2,3,1)
        dist = Categorical(x)
        y = dist.sample()
        y = y.cuda()
        return y/M
    
def save_display(x,path=None):
    grid = torchvision.utils.make_grid(x,nrow=10,padding=1)
    if path:
        torchvision.utils.save_image(grid,path)
    else:
        torchvision.utils.save_image(grid,'tmp.jpg')

if __name__ == '__main__':
    #run ```python util.py```  to download for the first time
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transformGrey)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transformGrey)
