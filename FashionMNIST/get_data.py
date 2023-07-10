from torchvision.datasets import FashionMNIST
import os
if __name__ == '__main__':
    train_set = FashionMNIST('../',train=True,download=True)
    test_set = FashionMNIST('../',train=False,download=True)
    os.system('rm -r raw')