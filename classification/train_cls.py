'''Train CIFAR10 with PyTorch.'''
import sys
sys.path.append("/home/shuxuang/eval-code/SSD")
# sys.path.append("/workspace/code/SSD")
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision

import os
import argparse
import shutil

# from models import *
# from cls_utils.utils import progress_bar
from tensorboardX import SummaryWriter
from data.load_data import make_data_loader_cls
from ssd.config import cfg_cls as cfg
import torchvision.models as models

import IPython


parser = argparse.ArgumentParser(description='PyTorch VOC classification Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--wd', default=4e-5, type=float, help='weight dacay')
parser.add_argument('--nepochs', default=150, type=int, help='learning rate')
parser.add_argument('--input_size', default=224, type=int, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--pretrained', default=False, type=bool, help='model init from model pretrained from ImageNet')
parser.add_argument('--arch', '-a', default='resnet18', type=str, 
                    help='mobilenetv1 | mobilenetv2 |mobilenetv1_expand | mobilenetv2_expand  ') 
parser.add_argument('--data', default=None, type=str, 
                    help='data for training and testing') 

parser.add_argument(
        "--config-file",
        default="resnet18_voc0712_cls.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )                  
args = parser.parse_args()
num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
args.distributed = num_gpus > 1
args.num_gpus = num_gpus

cfg.merge_from_file(args.config_file)
cfg.INPUT.IMAGE_SIZE = args.input_size
cfg.freeze()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
best_epoch = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

data_dirs = {
    'original': '/dataset/',
    # 'original': '/home/shuxuang/data/voc-cls/'
}
data_dir = data_dirs[args.data]
# Data
print('==> Preparing data..')
trainloader = make_data_loader_cls(cfg, data_dir, is_train=True, distributed=args.distributed, start_iter=0)
testloader = make_data_loader_cls(cfg, data_dir, is_train=False, distributed=args.distributed)

save_path = os.path.join("./cls_results", args.data, args.arch + '_input_' + str(cfg.INPUT.IMAGE_SIZE) + '_pretrained_' + str(args.pretrained))

# if os.path.exists(save_path):
#     shutil.rmtree(save_path)
if not os.path.isdir(save_path):
    os.makedirs(save_path)

Logwriter = SummaryWriter(save_path)
log = os.path.join(save_path, 'log.txt')

logfile = open(log, 'a')
logfile.write("Log_save_path: " + log + "\n" + "Configuration: " + str(args) + "\n" + str(cfg) + "\n")
logfile.close()

# Model
print('==> Building model..')
# net = build_models(args.arch, 21)
if args.arch == 'resnet50':
    net = models.resnet50(pretrained=args.pretrained)
    net.fc = nn.Linear(2048, 20)
elif args.arch == 'resnet18':
    net = models.resnet18(pretrained=args.pretrained)
    net.fc = nn.Linear(512, 20)
# IPython.embed()
print(net)


logfile = open(log, 'a')
logfile.write(str(net))
logfile.close()

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(save_path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(save_path, 'ckpt.pth'))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=args.wd) # 4e-5


# Training
def train(epoch):
    adjust_learning_rate(optimizer, args.lr, epoch)
    print('\nEpoch: %d, lr: %.4f' % (epoch,optimizer.param_groups[0]['lr']))
    logfile = open(log, 'a')
    logfile.write('\nEpoch: %d, lr: %.4f' % (epoch,optimizer.param_groups[0]['lr']))
    logfile.close()
    Logwriter.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
    net.train()
    train_loss = 0.
    correct = 0.
    total = 0.
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
    logfile = open(log, 'a')
    logfile.write('\nEpoch Tr: %d Loss: %.3f | Acc: %.3f%% (%d/%d)'% (epoch, train_loss/len(trainloader), 100.*correct/total, correct, total))
    logfile.close()
    Logwriter.add_scalar('Epoch_train_loss', train_loss/len(trainloader), epoch)
    Logwriter.add_scalar('Epoch_train_acc', 100.*correct/total, epoch)


def test(epoch):
    global best_acc
    global best_epoch
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        # IPython.embed()
        for batch_idx, (inputs, targets) in enumerate(testloader[0]):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader[0]), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #          % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

        logfile = open(log, 'a')
        logfile.write('\nEpoch Te: %d Loss: %.3f | Acc: %.3f%% (%d/%d)'% (epoch, test_loss/len(trainloader), 100.*correct/total, correct, total))
        logfile.close()
        Logwriter.add_scalar('Epoch_test_loss', test_loss/len(testloader), epoch)
        Logwriter.add_scalar('Epoch_test_acc', 100.*correct/total, epoch)

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        torch.save(state, save_path + '/ckpt.pth')
        best_acc = acc
        best_epoch = epoch


def adjust_learning_rate(optimizer, init_lr, epoch):
    if epoch < 50:
        lr = init_lr      
    else:
        if epoch < 100:
            lr = init_lr * 0.1 # * (0.1 ** (epoch // 200))
        else:  
            lr = init_lr * 0.01  # (0.1 ** (epoch // 300))
        
    for param_group in optimizer.param_groups:
            param_group['lr'] = lr


for epoch in range(start_epoch, args.nepochs):
    train(epoch)
    test(epoch)
    print('Best acc: %.4f '% best_acc)
    Logwriter.add_scalar('Best_acc', best_acc,  best_epoch)
    logfile = open(log, 'a')
    logfile.write('\nBest acc: %.4f '% best_acc)
    logfile.close()

# test only


# python train_cls.py --pretrained True --lr 0.001 --data original
# python train_cls.py --pretrained True --lr 0.001 --data noise1
# python train_cls.py --pretrained True --lr 0.001 --data noise10

# python train_cls.py --pretrained True --lr 0.001 --data noise1 --resume