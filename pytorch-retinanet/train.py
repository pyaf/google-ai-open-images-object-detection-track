from __future__ import print_function

import os
import argparse
import pdb
import traceback

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from loss import FocalLoss
from retinanet import RetinaNet
from dataloader import provider

from torch.autograd import Variable


parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

assert torch.cuda.is_available(), 'Error: CUDA not found!'
best_loss = float('inf')  # best val loss
start_epoch = 0  # start from epoch 0 or last epoch

# Model
print('==> Preparing Model..')
net = RetinaNet()
net.load_state_dict(torch.load('./model/net.pth'))
if args.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']

net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
net.cuda()

criterion = FocalLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    trainloader = provider(mode='train', batch_size=6)
    net.train()
    net.module.freeze_bn()
    train_loss = 0
    total_iterations = len(trainloader)
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(trainloader):
        inputs = Variable(inputs.cuda())
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())

        optimizer.zero_grad()
        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        print('train_loss: %.3f | avg_loss: %.3f | %d/%d' % (
            loss.item(),
            train_loss/(batch_idx+1),
            batch_idx,
            total_iterations)
        )
    trainloader = None

# val
def val(epoch):
    print('\nval')
    valloader = provider(mode='val', batch_size=4)
    net.eval()
    val_loss = 0
    total_iterations = len(valloader)
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(valloader):
        inputs = Variable(inputs.cuda())
        loc_targets = Variable(loc_targets.cuda())
        cls_targets = Variable(cls_targets.cuda())

        loc_preds, cls_preds = net(inputs)
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        val_loss += loss.item()
        print('val_loss: %.3f | avg_loss: %.3f | %d/%d' % (
            loss.item(),
            val_loss/(batch_idx+1),
            batch_idx,
            total_iterations)
        )

    # Save checkpoint
    global best_loss
    val_loss /= len(valloader)
    if val_loss < best_loss:
        print('Saving..')
        state = {
            'net': net.module.state_dict(),
            'loss': val_loss,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/model%d.pth' % epoch)
        best_loss = val_loss
    valloader = None

def save_state(epoch):
    state = {
            'net': net.module.state_dict(),
            'epoch': epoch,
        }
    torch.save(state, './checkpoint/ckpt%d.pth' % epoch)


for epoch in range(start_epoch, start_epoch+20):
    try:
        train(epoch)
        save_state(epoch)
        val(epoch)
    except KeyboardInterrupt as e:
        print(e)
        save_state(epoch)
        traceback.print_exc()
        pdb.set_trace()
    except Exception as e:
        traceback.print_exc()
        pdb.set_trace()
