from utils import *
from config import *
from network import PixelCNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

net = PixelCNN(n_layer,h_dim,binary_flag=binary_flag).cuda()
tmp_dict = torch.load(f'checkpoint/e{n_epoch}.pt')
net.load_state_dict(tmp_dict)
net.eval()

if binary_flag == True:
    #Reconstruct
    test_loader = testLoader(bs=batch_size)
    for data,label in test_loader:
        data = data.cuda()
        input = binarize(data).cuda()
        save_display(input,'sample/input.jpg')
        output = net(input)
        save_display(output,'sample/output.jpg')
        break

    #Generate via random sampling
    x = torch.zeros(data.shape).cuda()
    for i in range(28):
        for j in range(28):
            input = binarize(x,sample=True)
            output = net(input).detach()
            x[:,:,i,j] = output[:,:,i,j]
    save_display(x,'sample/generate.jpg')
else:
    #Reconstruct
    test_loader = testLoader(bs=batch_size)
    for data,label in test_loader:
        data = data.cuda()
        input = data
        save_display(input,'sample/input.jpg')
        output = net(input)
        output = torch.argmax(output,dim=1) / 255.
        output = output.unsqueeze(1)
        save_display(output,'sample/output.jpg')
        break
    #Generate via random sampling
    x = torch.zeros(data.shape).cuda()
    for i in range(28):
        for j in range(28):
            output = net(x).detach()
            output = F.softmax(output)
            output = quantize(output,sample=True).unsqueeze(1)
            x[:,:,i,j] = output[:,:,i,j]
    save_display(x,'sample/generate.jpg')