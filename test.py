from utils import *
from config import *
from network import PixelCNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

net = PixelCNN(n_layer,h_dim,k_size,color_bit).cuda()
tmp_dict = torch.load(f'checkpoint/e{n_epoch}.pt')
net.load_state_dict(tmp_dict)
net.eval()

#Reconstruct
test_loader = testLoader(bs=batch_size)
for data,label in test_loader:
    data = data.cuda()
    input = data
    save_display(input,'sample/input.jpg')
    output = net(input)
    output = torch.argmax(output,dim=1) / (2**color_bit-1)
    output = output.unsqueeze(1)
    save_display(output,'sample/output.jpg')
    break

#Generate via random sampling
x = torch.zeros(data.shape).cuda()
for i in range(28):
    print(f'{i+1}/28')
    for j in range(28):
        output = net(x).detach()
        output = F.softmax(output)
        output = quantize(output,bit=color_bit,sample=True).unsqueeze(1)
        x[:,:,i,j] = output[:,:,i,j]
save_display(x,'sample/generate.jpg')