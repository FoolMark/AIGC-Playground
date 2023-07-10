from torchvision.datasets import FashionMNIST as A
from torchvision.datasets import MNIST as B
from torchvision.datasets import CIFAR10 as C
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
import os

def Tran(size=-1):
    if size <= 0:
        tran = T.Compose([
            T.ToTensor()
        ])
    else:
        tran = T.Compose([
            T.Resize(size=size),
            T.ToTensor()
        ])
    return tran

def FashionMNIST(size):
    cur = os.path.dirname(__file__)
    train_set= A(cur,train=True,transform=Tran())
    test_set= A(cur,train=False,transform=Tran())
    return train_set,test_set

def MNIST(size):
    cur = os.path.dirname(__file__)
    train_set= B(cur,train=True,transform=Tran())
    test_set= B(cur,train=False,transform=Tran())
    return train_set,test_set

def CIFAR10(size):
    cur = os.path.join(os.path.dirname(__file__),'CIFAR10')
    train_set= C(cur,train=True,transform=Tran())
    test_set= C(cur,train=False,transform=Tran())
    return train_set,test_set

def ANIME(size=64):
    cur = os.path.join(os.path.dirname(__file__),'Anime')
    dset = ImageFolder(root=cur,transform=Tran(size=size))
    return dset,None

def CAT(size=128):
    cur = os.path.join(os.path.dirname(__file__),'Cat')
    dset = ImageFolder(root=cur,transform=Tran(size=size))
    return dset,None

def getData(name,*args):
    if name == 'CAT':
        return CAT(*args)
    if name == 'ANIME':
        return ANIME(*args)
    if name == 'CIFAR10':
        return CIFAR10(*args)
    if name == 'MNIST':
        return MNIST(*args)
    if name == 'FashionMNIST':
        return FashionMNIST(*args)