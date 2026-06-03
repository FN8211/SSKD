"""
student_rot.py — Rotation SSKD for Tiny ImageNet (vgg13 → vgg8).
"""
import os
import os.path as osp
import sys
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
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
from tiny_imagenet import TinyImageNet, TIN_MEAN, TIN_STD

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Tiny ImageNet SSKD student (Rotation).')
parser.add_argument('--epoch',        type=int,   default=200)
parser.add_argument('--t-epoch',      type=int,   default=60)
parser.add_argument('--batch-size',   type=int,   default=64)
parser.add_argument('--val-interval', type=int,   default=1)
parser.add_argument('--lr',           type=float, default=0.05)
parser.add_argument('--t-lr',         type=float, default=0.05)
parser.add_argument('--momentum',     type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--gamma',        type=float, default=0.1)
parser.add_argument('--milestones',   type=int,   nargs='+', default=[100, 150, 180])
parser.add_argument('--t-milestones', type=int,   nargs='+', default=[30, 45])
parser.add_argument('--save-interval',type=int,   default=40)
parser.add_argument('--ce-weight',    type=float, default=0.1)
parser.add_argument('--kd-weight',    type=float, default=0.9)
parser.add_argument('--tf-weight',    type=float, default=2.7)
parser.add_argument('--ss-weight',    type=float, default=10.0)
parser.add_argument('--kd-T',         type=float, default=4.0)
parser.add_argument('--tf-T',         type=float, default=4.0)
parser.add_argument('--ss-T',         type=float, default=0.5)
parser.add_argument('--ratio-tf',     type=float, default=1.0)
parser.add_argument('--s-arch',       type=str,   default='vgg8')
parser.add_argument('--t-path',       type=str,   required=True)
parser.add_argument('--data-root',    type=str,   default='./data/tiny-imagenet-200')
parser.add_argument('--seed',         type=int,   default=0)
parser.add_argument('--gpu-id',       type=int,   default=0)
args = parser.parse_args()


def _parse_t_arch(t_path):
    """tin_teacher_{arch}  →  {arch}"""
    t_name = osp.basename(osp.normpath(t_path))
    parts = t_name.split('_')
    return '_'.join(parts[2:])


class wrapper_rotation(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.backbone = module
        self.proj_head = nn.Linear(module.classifier.in_features, 4)

    def forward(self, x, bb_grad=True):
        if bb_grad:
            feats, logit = self.backbone(x, is_feat=True)
        else:
            with torch.no_grad():
                feats, logit = self.backbone(x, is_feat=True)
        return logit, self.proj_head(feats[-1]), feats[-1]


if __name__ == '__main__':
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    t_name = osp.basename(osp.normpath(args.t_path))
    t_arch = _parse_t_arch(args.t_path)

    exp_name = 'tin_sskd_rot_student_{}_weight{}+{}+{}+{}_T{}+{}+{}_ratio{}_{}'.format(
        args.s_arch,
        args.ce_weight, args.kd_weight, args.tf_weight, args.ss_weight,
        args.kd_T, args.tf_T, args.ss_T,
        args.ratio_tf, t_name)
    exp_path = osp.join(osp.dirname(__file__), '..', 'experiments_tin', exp_name)
    os.makedirs(exp_path, exist_ok=True)

    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.ToTensor(),
        transforms.Normalize(mean=TIN_MEAN, std=TIN_STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=TIN_MEAN, std=TIN_STD),
    ])

    trainset = TinyImageNet(args.data_root, train=True,  transform=transform_train)
    valset   = TinyImageNet(args.data_root, train=False, transform=transform_test)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(valset,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=False)

    ckpt_path = osp.join(args.t_path, 'ckpt', 'best.pth')
    t_model = model_dict[t_arch](num_classes=200).cuda()
    t_model.load_state_dict(torch.load(ckpt_path, weights_only=False)['state_dict'])
    t_model = wrapper_rotation(module=t_model).cuda()

    t_optimizer = optim.SGD(
        [{'params': t_model.backbone.parameters(), 'lr': 0.0},
         {'params': t_model.proj_head.parameters(), 'lr': args.t_lr}],
        momentum=args.momentum, weight_decay=args.weight_decay)
    t_scheduler = MultiStepLR(t_optimizer, milestones=args.t_milestones, gamma=args.gamma)
    logger = SummaryWriter(osp.join(exp_path, 'events'))

    # ── quick check: teacher classification accuracy ──
    t_model.eval()
    acc_record = AverageMeter()
    for x, target in tqdm(val_loader):
        x, target = x[:, 0].cuda(), target.cuda()
        with torch.no_grad():
            output, _, _ = t_model(x)
        acc_record.update(accuracy(output, target, topk=(1,))[0].item(), x.size(0))
    print('teacher cls_acc:{:.2f}\n'.format(acc_record.avg))

    # ── Stage 1: train teacher rotation head (backbone frozen) ──
    for epoch in range(args.t_epoch):
        t_model.eval()
        loss_record = AverageMeter()
        acc_record  = AverageMeter()
        start = time.time()
        for x, _ in tqdm(train_loader):
            t_optimizer.zero_grad()
            c, h, w = x.size()[-3:]
            x_rot = x.view(-1, c, h, w).cuda()                # (4N, C, H, W)
            rot_labels = torch.arange(4).unsqueeze(0).expand(
                x.size(0), -1).reshape(-1).long().cuda()
            _, rot_logit, _ = t_model(x_rot, bb_grad=False)
            loss = F.cross_entropy(rot_logit, rot_labels)
            loss.backward()
            t_optimizer.step()
            loss_record.update(loss.item(), x_rot.size(0))
            acc_record.update(accuracy(rot_logit, rot_labels, topk=(1,))[0].item(), x_rot.size(0))

        logger.add_scalar('train/teacher_ssp_loss', loss_record.avg, epoch + 1)
        logger.add_scalar('train/teacher_ssp_acc',  acc_record.avg,  epoch + 1)
        print('teacher_train_Epoch:{:03d}/{:03d}\t run_time:{:.1f}s\t '
              'ssp_loss:{:.3f}\t ssp_acc:{:.2f}'.format(
              epoch + 1, args.t_epoch, time.time() - start,
              loss_record.avg, acc_record.avg))

        if (epoch + 1) % args.val_interval == 0:
            t_model.eval()
            acc_record  = AverageMeter()
            loss_record = AverageMeter()
            for x, _ in tqdm(val_loader):
                c, h, w = x.size()[-3:]
                x_rot = x.view(-1, c, h, w).cuda()
                rot_labels = torch.arange(4).unsqueeze(0).expand(
                    x.size(0), -1).reshape(-1).long().cuda()
                with torch.no_grad():
                    _, rot_logit, _ = t_model(x_rot)
                loss = F.cross_entropy(rot_logit, rot_labels)
                loss_record.update(loss.item(), x_rot.size(0))
                acc_record.update(accuracy(rot_logit, rot_labels, topk=(1,))[0].item(), x_rot.size(0))
            logger.add_scalar('val/teacher_ssp_loss', loss_record.avg, epoch + 1)
            logger.add_scalar('val/teacher_ssp_acc',  acc_record.avg,  epoch + 1)
            print('ssp_test_Epoch:{:03d}/{:03d}\t ssp_loss:{:.3f}\t ssp_acc:{:.2f}\n'.format(
                  epoch + 1, args.t_epoch, loss_record.avg, acc_record.avg))

        t_scheduler.step()

    name = osp.join(exp_path, 'ckpt', 'teacher.pth')
    os.makedirs(osp.dirname(name), exist_ok=True)
    torch.save(t_model.state_dict(), name)

    # ── Stage 2: train student ──
    s_model = wrapper_rotation(module=model_dict[args.s_arch](num_classes=200)).cuda()
    optimizer = optim.SGD(s_model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    best_acc = 0
    for epoch in range(args.epoch):
        s_model.train()
        loss1_record  = AverageMeter()
        loss2_record  = AverageMeter()
        loss3_record  = AverageMeter()
        loss4_record  = AverageMeter()
        cls_acc_record = AverageMeter()
        ssp_acc_record = AverageMeter()
        start = time.time()

        for x, target in tqdm(train_loader):
            optimizer.zero_grad()
            target = target.cuda()
            batch  = x.size(0)
            c, h, w = x.size()[-3:]

            x_nor = x[:, 0].cuda()
            x_rot = x.view(-1, c, h, w).cuda()
            rot_labels = torch.arange(4).unsqueeze(0).expand(
                batch, -1).reshape(-1).long().cuda()

            s_output,  s_rot_logit,  _ = s_model(x_nor, bb_grad=True)
            s_rot_out, s_rot_logit2, _ = s_model(x_rot, bb_grad=True)

            with torch.no_grad():
                t_output,  _,            _ = t_model(x_nor)
                t_rot_out, t_rot_logit2, _ = t_model(x_rot)

            loss1 = F.cross_entropy(s_output, target)

            log_s_norm   = F.log_softmax(s_output   / args.kd_T, dim=1)
            t_norm_soft  = F.softmax(t_output / args.kd_T, dim=1).detach()
            loss2 = F.kl_div(log_s_norm, t_norm_soft, reduction='batchmean') * args.kd_T ** 2

            aug_knowledge  = F.softmax(t_rot_out / args.tf_T, dim=1).detach()
            log_aug_output = F.log_softmax(s_rot_out / args.tf_T, dim=1)
            aug_target_cls = target.unsqueeze(1).expand(-1, 4).reshape(-1)
            rank = torch.argsort(aug_knowledge, dim=1, descending=True)
            rank = torch.argmax(torch.eq(rank, aug_target_cls.unsqueeze(1)).long(), dim=1)
            index = torch.argsort(rank)
            wrong_num   = torch.nonzero(rank, as_tuple=True)[0].numel()
            correct_num = 4 * batch - wrong_num
            index = index[:correct_num + int(wrong_num * args.ratio_tf)]
            distill_index_tf = torch.sort(index)[0]
            loss3 = F.kl_div(log_aug_output[distill_index_tf],
                             aug_knowledge[distill_index_tf],
                             reduction='batchmean') * args.tf_T ** 2

            t_rot_soft = F.softmax(t_rot_logit2 / args.ss_T, dim=1).detach()
            log_s_rot  = F.log_softmax(s_rot_logit2 / args.ss_T, dim=1)
            loss4 = F.kl_div(log_s_rot, t_rot_soft, reduction='batchmean') * args.ss_T ** 2

            loss = (args.ce_weight * loss1 + args.kd_weight * loss2
                  + args.tf_weight * loss3 + args.ss_weight * loss4)
            loss.backward()
            optimizer.step()

            cls_acc_record.update(accuracy(s_output,     target,     topk=(1,))[0].item(), batch)
            ssp_acc_record.update(accuracy(s_rot_logit2, rot_labels, topk=(1,))[0].item(), 4 * batch)
            loss1_record.update(loss1.item(), batch)
            loss2_record.update(loss2.item(), batch)
            loss3_record.update(loss3.item(), len(distill_index_tf))
            loss4_record.update(loss4.item(), 4 * batch)

        logger.add_scalar('train/ce_loss',  loss1_record.avg,  epoch + 1)
        logger.add_scalar('train/kd_loss',  loss2_record.avg,  epoch + 1)
        logger.add_scalar('train/tf_loss',  loss3_record.avg,  epoch + 1)
        logger.add_scalar('train/ss_loss',  loss4_record.avg,  epoch + 1)
        logger.add_scalar('train/cls_acc',  cls_acc_record.avg, epoch + 1)
        logger.add_scalar('train/ss_acc',   ssp_acc_record.avg, epoch + 1)
        print('student_train_Epoch:{:03d}/{:03d}\t run_time:{:.1f}s\t '
              'ce_loss:{:.3f}\t kd_loss:{:.3f}\t cls_acc:{:.2f}'.format(
              epoch + 1, args.epoch, time.time() - start,
              loss1_record.avg, loss2_record.avg, cls_acc_record.avg))

        if (epoch + 1) % args.val_interval == 0:
            s_model.eval()
            acc_record  = AverageMeter()
            loss_record = AverageMeter()
            for x, target in tqdm(val_loader):
                x, target = x[:, 0].cuda(), target.cuda()
                with torch.no_grad():
                    output, _, _ = s_model(x)
                    loss = F.cross_entropy(output, target)
                acc_record.update(accuracy(output, target, topk=(1,))[0].item(), x.size(0))
                loss_record.update(loss.item(), x.size(0))
            logger.add_scalar('val/ce_loss',  loss_record.avg, epoch + 1)
            logger.add_scalar('val/cls_acc',  acc_record.avg,  epoch + 1)
            print('student_test_Epoch:{:03d}/{:03d}\t cls_acc:{:.2f}\n'.format(
                  epoch + 1, args.epoch, acc_record.avg))

            if acc_record.avg > best_acc:
                best_acc = acc_record.avg
                state = dict(epoch=epoch + 1,
                             state_dict=s_model.state_dict(),
                             best_acc=best_acc)
                name = osp.join(exp_path, 'ckpt', 'student_best.pth')
                os.makedirs(osp.dirname(name), exist_ok=True)
                torch.save(state, name)

        scheduler.step()
