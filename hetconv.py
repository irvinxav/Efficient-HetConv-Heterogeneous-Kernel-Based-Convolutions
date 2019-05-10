import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

parser = argparse.ArgumentParser(description='CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg16')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=330, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=200, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
best_prec1 = 0

################################################################################
# Implementation of HetConv using group wise and point wise convolution

class HetConv(nn.Module):
    def __init__(self, in_channels, out_channels, p):
        super(HetConv, self).__init__()
        # Groupwise Convolution
        self.gwc = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=p, bias=False)
        # Pointwise Convolution
        self.pwc = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.gwc(x) + self.pwc(x)
################################################################################

class vgg16bn(nn.Module):
    def __init__(self,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,p):
        super(vgg16bn, self).__init__()    
        
        self.features = nn.Sequential(
            nn.Conv2d(3, f1, kernel_size=3, padding=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            HetConv(f1, f2, p),           
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            HetConv(f2, f3, p),
            nn.BatchNorm2d(f3),
            nn.ReLU(inplace=True), 
            HetConv(f3, f4, p),
            nn.BatchNorm2d(f4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            HetConv(f4, f5, p),
            nn.BatchNorm2d(f5),
            nn.ReLU(inplace=True), 
            HetConv(f5, f6, p),
            nn.BatchNorm2d(f6),
            nn.ReLU(inplace=True),
            HetConv(f6, f7, p),
            nn.BatchNorm2d(f7),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            HetConv(f7, f8, p),
            nn.BatchNorm2d(f8),
            nn.ReLU(inplace=True),
            HetConv(f8, f9, p),
            nn.BatchNorm2d(f9),
            nn.ReLU(inplace=True),
            HetConv(f9, f10, p),
            nn.BatchNorm2d(f10),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            HetConv(f10, f11, p),
            nn.BatchNorm2d(f11),
            nn.ReLU(inplace=True),
            HetConv(f11, f12, p),
            nn.BatchNorm2d(f12),
            nn.ReLU(inplace=True),
            HetConv(f12, f13, p),
            nn.BatchNorm2d(f13),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.classifier = nn.Sequential(            
            nn.Linear(512*1*1, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),                        
            nn.Linear(512, 10),
        )
              
        # UNFreeze those weights
        for p in self.features.parameters():
            p.requires_grad = True
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x               

def main():
    global args, best_prec1
    args = parser.parse_args()

    part = 4 # By changing "part" P value, You can reproduce the results for VGG-16 on CIFAR-10.
    
    model = vgg16bn(64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512, part)    
    model = model.cuda()

    #model.load_state_dict(torch.load("XXXYYY.pth"))

    for p in model.parameters():
        p.requires_grad = True   

    cudnn.benchmark = True

################################################################################
# Code to make corresponding extra M/P 1x1 kernels weights to zero and masking the corresponding gradients so that extra M/P 1x1 kernels weights remains zero during backpropagations.

    convlist = [model.features[3],model.features[7],model.features[10],model.features[14],model.features[17],model.features[20],model.features[24],model.features[27],model.features[30],model.features[34],model.features[37],model.features[40]]
    mask = [[] for y in range(12)]
    m = 0
    for layer in convlist:
        gp = part
        layerw = layer.pwc.weight.data.cpu().numpy()
        wtmask=np.ones(layerw.shape)
        tf, fl,_ ,_ = layerw.shape
        gps = int(fl/gp)
        Nfilt=int(tf/gp)
        j=0
        k=0        
        for i in range(gp):
            layerw[k:k+Nfilt,j:j+gps,:,:] = 0
            wtmask[k:k+Nfilt,j:j+gps,:,:] = 0   
            j=j+gps
            k=k+Nfilt
        layer.pwc.weight.data = (torch.FloatTensor(layerw).cuda())        
        layermask = (torch.FloatTensor(wtmask).cuda())
        mask[m] = layermask
        m = m + 1
################################################################################

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            #transforms.Resize((32, 32)),
            transforms.ToTensor(),
            #normalize,
        ]), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            #transforms.Resize((32, 32)),
            transforms.ToTensor(),
            #normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()
    
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[50, 140, 210, 240, 270, 300], gamma=0.2, last_epoch=args.start_epoch - 1)
    
    best = validate(val_loader, model, criterion)

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch, convlist, mask)
        lr_scheduler.step()

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        if prec1 > best:
            fname = 'HetConv_'+str(prec1)+'.pth'
            torch.save(model.state_dict(), fname)
            best = prec1


def train(train_loader, model, criterion, optimizer, epoch, convlist, mask):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target)
        
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
################################################################################
# Code for masking the corresponding gradients so that extra M/P 1x1 kernels weights remains zero during backpropagations.

        m = 0
        for layer in convlist:
            for p in layer.pwc.parameters():
                p.grad *= mask[m] # print(p.grad)
                m = m + 1
                break

################################################################################

        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
