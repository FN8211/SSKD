import os
import os.path as osp
import sys
import argparse
import time
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, osp.join(osp.dirname(__file__), '..'))
from utils import AverageMeter, accuracy
from models import model_dict
from tiny_imagenet import TinyImageNetPlain, TIN_MEAN, TIN_STD

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='train Tiny ImageNet teacher (vgg13).')
parser.add_argument('--epoch',        type=int,   default=240)
parser.add_argument('--batch-size',   type=int,   default=128)
parser.add_argument('--val-interval', type=int,   default=1)
parser.add_argument('--lr',           type=float, default=0.05)
parser.add_argument('--momentum',     type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--gamma',        type=float, default=0.1)
parser.add_argument('--milestones',   type=int,   nargs='+', default=[150, 180, 210])
parser.add_argument('--save-interval',type=int,   default=40)
parser.add_argument('--arch',         type=str,   default='vgg13')
parser.add_argument('--data-root',    type=str,   default='./data/tiny-imagenet-200')
parser.add_argument('--seed',         type=int,   default=0)
parser.add_argument('--gpu-id',       type=int,   default=0)
args = parser.parse_args()

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    exp_name = 'tin_teacher_{}'.format(args.arch)
    exp_path = osp.join(osp.dirname(__file__), '..', 'experiments_tin', exp_name)
    os.makedirs(exp_path, exist_ok=True)

    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=TIN_MEAN, std=TIN_STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=TIN_MEAN, std=TIN_STD),
    ])

    trainset = TinyImageNetPlain(args.data_root, train=True,  transform=transform_train)
    valset   = TinyImageNetPlain(args.data_root, train=False, transform=transform_test)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(valset,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=False)

    model = model_dict[args.arch](num_classes=200).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    logger = SummaryWriter(osp.join(exp_path, 'events'))

    best_acc = -1
    for epoch in range(args.epoch):
        model.train()
        loss_record = AverageMeter()
        acc_record  = AverageMeter()
        start = time.time()
        for x, target in tqdm(train_loader):
            optimizer.zero_grad()
            x, target = x.cuda(), target.cuda()
            output = model(x)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            batch_acc = accuracy(output, target, topk=(1,))[0]
            loss_record.update(loss.item(), x.size(0))
            acc_record.update(batch_acc.item(), x.size(0))

        logger.add_scalar('train/cls_loss', loss_record.avg, epoch + 1)
        logger.add_scalar('train/cls_acc',  acc_record.avg,  epoch + 1)
        print('train_Epoch:{:03d}/{:03d}\t run_time:{:.1f}s\t '
              'cls_loss:{:.3f}\t cls_acc:{:.2f}'.format(
              epoch + 1, args.epoch, time.time() - start,
              loss_record.avg, acc_record.avg))

        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            acc_record  = AverageMeter()
            loss_record = AverageMeter()
            start = time.time()
            for x, target in tqdm(val_loader):
                x, target = x.cuda(), target.cuda()
                with torch.no_grad():
                    output = model(x)
                    loss   = F.cross_entropy(output, target)
                batch_acc = accuracy(output, target, topk=(1,))[0]
                loss_record.update(loss.item(), x.size(0))
                acc_record.update(batch_acc.item(), x.size(0))

            logger.add_scalar('val/cls_loss', loss_record.avg, epoch + 1)
            logger.add_scalar('val/cls_acc',  acc_record.avg,  epoch + 1)
            print('test_Epoch:{:03d}/{:03d}\t run_time:{:.1f}s\t '
                  'cls_loss:{:.3f}\t cls_acc:{:.2f}\n'.format(
                  epoch + 1, args.epoch, time.time() - start,
                  loss_record.avg, acc_record.avg))

            if acc_record.avg > best_acc:
                best_acc = acc_record.avg
                state_dict = dict(epoch=epoch + 1,
                                  state_dict=model.state_dict(),
                                  acc=acc_record.avg)
                name = osp.join(exp_path, 'ckpt', 'best.pth')
                os.makedirs(osp.dirname(name), exist_ok=True)
                torch.save(state_dict, name)

        scheduler.step()

    print('best_acc: {:.2f}'.format(best_acc))
