from torchvision.datasets import MNIST
import os
if __name__ == '__main__':
    train_set = MNIST('../',train=True,download=True)
    test_set = MNIST('../',train=False,download=True)
    os.system('rm -r raw')