from utils import *
from config import *
from network import PixelCNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

logger,tbWriter = initial()
logger.info(f'N layer: {n_layer} H dim: {h_dim}')
train_loader = trainLoader(bs=batch_size)
test_loader = testLoader(bs=batch_size)



net = PixelCNN(n_layer,h_dim,binary_flag=binary_flag).cuda()
if opt_type == 'sgd':
    opt = optim.SGD(lr=base_lr,weight_decay=weight_decay,params=net.parameters())
else:
    opt = optim.AdamW(lr=base_lr,weight_decay=weight_decay,params=net.parameters())
if binary_flag:
    loss_func = nn.BCELoss()
else:
    loss_func = nn.CrossEntropyLoss()

iter = 0
for epoch in range(n_epoch):
    net.train()
    cur_lr = adjust_lr(opt,base_lr,epoch,n_epoch)
    logger.info(f'Current lr: {cur_lr}')
    for data,label in train_loader:
        iter += 1
        data = data.cuda()
        if binary_flag:
            input = binarize(data)
        else:
            input = data
        output = net(input)
        if binary_flag:
            target = binarize(data).detach()
        else:
            target = quantize(data).detach()
            target = target.squeeze(1).long()

        loss = loss_func(output,target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        logger.info(f'[Train] Epoch {epoch+1}/{n_epoch}, Iter {iter}, Loss: {loss.item()}')
        if iter % 100 == 0:
            print(f'[Train] Epoch {epoch+1}/{n_epoch}, Iter {iter}, Loss: {loss.item()}')
        tbWriter.add_scalar('train_loss',loss.item(),iter)
    net.eval()
    test_loss = 0.0
    test_count = 0
    with torch.no_grad():
        for data,label in test_loader:
            data = data.cuda()
            if binary_flag:
                input = binarize(data)
            else:
                input = data
            output = net(input)
            if binary_flag:
                target = binarize(data).detach()
            else:
                target = quantize(data).detach()
                target = target.squeeze(1).long()
            loss = loss_func(output,target)
            test_loss += loss.item()
            test_count += 1
    test_loss /= test_count
    logger.info(f'[Test] Epoch {epoch+1}/{n_epoch}, Loss: {test_loss}')
    print(f'[Test] Epoch {epoch+1}/{n_epoch}, Loss: {test_loss}')
    tbWriter.add_scalar('test_loss',test_loss,iter)
    if (epoch+1) % 10 == 0:
        torch.save(net.state_dict(),f'checkpoint/e{epoch+1}.pt')
