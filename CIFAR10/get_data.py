from torchvision.datasets import CIFAR10
import os
if __name__ == '__main__':
    train_set = CIFAR10('./',train=True,download=True)
    test_set = CIFAR10('./',train=False,download=True)
    os.system('rm *.tar.gz')