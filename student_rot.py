import os
import os.path as osp
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

from utils import AverageMeter, accuracy
from cifar import CIFAR100
from tqdm import tqdm

from models import model_dict

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='train SSKD student network (Rotation).')
parser.add_argument('--epoch', type=int, default=240)
parser.add_argument('--t-epoch', type=int, default=60)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--val-interval', type=int, default=1)

parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--t-lr', type=float, default=0.05)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--milestones', type=int, nargs='+', default=[150, 180, 210])
parser.add_argument('--t-milestones', type=int, nargs='+', default=[30, 45])

parser.add_argument('--save-interval', type=int, default=40)
parser.add_argument('--ce-weight', type=float, default=0.1)
parser.add_argument('--kd-weight', type=float, default=0.9)
parser.add_argument('--tf-weight', type=float, default=2.7)
parser.add_argument('--ss-weight', type=float, default=10.0)

parser.add_argument('--kd-T', type=float, default=4.0)
parser.add_argument('--tf-T', type=float, default=4.0)
parser.add_argument('--ss-T', type=float, default=0.5)

# ratio-tf kept for transformed image KD (loss3), ratio-ss removed (no ranking for rotation)
parser.add_argument('--ratio-tf', type=float, default=1.0)
parser.add_argument('--s-arch', type=str)
parser.add_argument('--t-path', type=str)

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu-id', type=int, default=0)

args = parser.parse_args()


def unpack_rotation_batch(x):
    """
    cifar.py __getitem__ already returns (N, 4, C, H, W):
      index 0 = 0deg, 1 = 90deg, 2 = 180deg, 3 = 270deg
    This function unpacks it into (4N, C, H, W) + rotation labels.
    """
    c, h, w = x.size()[-3:]
    x_rot = x.view(-1, c, h, w)                # (4N, C, H, W)
    rot_labels = torch.arange(4).unsqueeze(0).expand(x.size(0), -1).reshape(-1).long()
    return x_rot, rot_labels


# ── wrapper_rotation: replaces the contrastive proj_head with a 4-way classifier ──
class wrapper_rotation(nn.Module):
    """
    Thin wrapper around a backbone that adds:
      - proj_head: 4-way rotation classifier (SS module)
    Uses is_feat=True to get intermediate features from the backbone.
    feat_dim is auto-detected from backbone.classifier.in_features.
    """
    def __init__(self, module):
        super().__init__()
        self.backbone = module
        feat_dim = list(module.children())[-1].in_features    # works for any backbone
        self.proj_head = nn.Linear(feat_dim, 4)

    def forward(self, x, bb_grad=True):
        if bb_grad:
            feats, logit = self.backbone(x, is_feat=True)
        else:
            with torch.no_grad():
                feats, logit = self.backbone(x, is_feat=True)
        feat = feats[-1]                            # last feature before classifier
        rot_logit = self.proj_head(feat)            # 4-way rotation logits
        return logit, rot_logit, feat


if __name__ == '__main__':
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    t_name = os.path.basename(os.path.normpath(args.t_path))
    t_arch = '_'.join(t_name.split('_')[1:])
    exp_name = 'sskd_rotation_student_{}_weight{}+{}+{}+{}_T{}+{}+{}_ratio{}_seed{}_{}'.format(
                args.s_arch,
                args.ce_weight, args.kd_weight, args.tf_weight, args.ss_weight,
                args.kd_T, args.tf_T, args.ss_T,
                args.ratio_tf,
                args.seed, t_name)
    exp_path = './experiments/{}'.format(exp_name)
    os.makedirs(exp_path, exist_ok=True)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4866, 0.4409],
                             std=[0.2675, 0.2565, 0.2761]),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4866, 0.4409],
                             std=[0.2675, 0.2565, 0.2761]),
    ])

    # CIFAR100 returns plain (img, label) — no multi-view augmentation needed
    trainset = CIFAR100('./data', train=True, transform=transform_train)
    valset   = CIFAR100('./data', train=False, transform=transform_test)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(valset,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=False)

    # ── load teacher backbone and wrap with rotation head ──
    ckpt_path = osp.join(args.t_path, 'ckpt/best.pth')
    t_model = model_dict[t_arch](num_classes=100).cuda()
    state_dict = torch.load(ckpt_path)['state_dict']
    t_model.load_state_dict(state_dict)
    t_model = wrapper_rotation(module=t_model).cuda()

    t_optimizer = optim.SGD(
        [{'params': t_model.backbone.parameters(), 'lr': 0.0},
         {'params': t_model.proj_head.parameters(), 'lr': args.t_lr}],
        momentum=args.momentum, weight_decay=args.weight_decay)
    t_scheduler = MultiStepLR(t_optimizer, milestones=args.t_milestones, gamma=args.gamma)

    logger = SummaryWriter(osp.join(exp_path, 'events'))

    # ── quick check: teacher classification accuracy ──
    t_model.eval()
    acc_record  = AverageMeter()
    loss_record = AverageMeter()
    start = time.time()
    for x, target in tqdm(val_loader):
        x, target = x[:,0,:,:,:].cuda(), target.cuda()
        with torch.no_grad():
            output, _, _ = t_model(x)
            loss = F.cross_entropy(output, target)
        batch_acc = accuracy(output, target, topk=(1,))[0]
        acc_record.update(batch_acc.item(), x.size(0))
        loss_record.update(loss.item(), x.size(0))
    print('teacher cls_acc:{:.2f}\n'.format(acc_record.avg))

    # ── Stage 1: train teacher rotation head (backbone frozen) ──
    for epoch in range(args.t_epoch):
        t_model.eval()          # backbone stays in eval; only proj_head trained
        loss_record = AverageMeter()
        acc_record  = AverageMeter()

        start = time.time()
        for x, _ in tqdm(train_loader):
            t_optimizer.zero_grad()

            x = x.cuda()
            x_rot, rot_labels = unpack_rotation_batch(x)   # (4N, C, H, W)
            rot_labels = rot_labels.cuda()

            _, rot_logit, _ = t_model(x_rot, bb_grad=False)
            loss = F.cross_entropy(rot_logit, rot_labels)

            loss.backward()
            t_optimizer.step()

            batch_acc = accuracy(rot_logit, rot_labels, topk=(1,))[0]
            loss_record.update(loss.item(), x_rot.size(0))
            acc_record.update(batch_acc.item(), x_rot.size(0))

        logger.add_scalar('train/teacher_ssp_loss', loss_record.avg, epoch + 1)
        logger.add_scalar('train/teacher_ssp_acc',  acc_record.avg,  epoch + 1)
        run_time = time.time() - start
        print('teacher_train_Epoch:{:03d}/{:03d}\t run_time:{:.3f}\t '
              'ssp_loss:{:.3f}\t ssp_acc:{:.2f}'.format(
              epoch + 1, args.t_epoch, run_time, loss_record.avg, acc_record.avg))

        if (epoch + 1) % args.val_interval == 0:
            t_model.eval()
            acc_record  = AverageMeter()
            loss_record = AverageMeter()
            start = time.time()
            for x, _ in tqdm(val_loader):
                x = x.cuda()
                x_rot, rot_labels = unpack_rotation_batch(x)
                rot_labels = rot_labels.cuda()
                with torch.no_grad():
                    _, rot_logit, _ = t_model(x_rot)
                loss = F.cross_entropy(rot_logit, rot_labels)
                batch_acc = accuracy(rot_logit, rot_labels, topk=(1,))[0]
                acc_record.update(batch_acc.item(), x_rot.size(0))
                loss_record.update(loss.item(), x_rot.size(0))

            run_time = time.time() - start
            logger.add_scalar('val/teacher_ssp_loss', loss_record.avg, epoch + 1)
            logger.add_scalar('val/teacher_ssp_acc',  acc_record.avg,  epoch + 1)
            print('ssp_test_Epoch:{:03d}/{:03d}\t run_time:{:.2f}\t '
                  'ssp_loss:{:.3f}\t ssp_acc:{:.2f}\n'.format(
                  epoch + 1, args.t_epoch, run_time, loss_record.avg, acc_record.avg))

        t_scheduler.step()

    name = osp.join(exp_path, 'ckpt/teacher.pth')
    os.makedirs(osp.dirname(name), exist_ok=True)
    torch.save(t_model.state_dict(), name)

    # ── Stage 2: train student ──
    s_model = model_dict[args.s_arch](num_classes=100)
    s_model = wrapper_rotation(module=s_model).cuda()
    optimizer = optim.SGD(s_model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    best_acc = 0
    for epoch in range(args.epoch):

        s_model.train()
        loss1_record = AverageMeter()
        loss2_record = AverageMeter()
        loss3_record = AverageMeter()
        loss4_record = AverageMeter()
        cls_acc_record = AverageMeter()
        ssp_acc_record = AverageMeter()

        start = time.time()
        for x, target in tqdm(train_loader):
            optimizer.zero_grad()

            target = target.cuda()
            batch = x.size(0)

            # x is (N, 4, C, H, W) from cifar.py
            # index 0 = normal image, 1-3 = rotated versions
            x_nor = x[:,0,:,:,:].cuda()                     # (N, C, H, W)
            x_rot, rot_labels = unpack_rotation_batch(x)    # (4N, C, H, W)
            x_rot      = x_rot.cuda()
            rot_labels = rot_labels.cuda()

            # ── student forward ──
            s_output,  s_rot_logit,  _ = s_model(x_nor, bb_grad=True)
            s_rot_out, s_rot_logit2, _ = s_model(x_rot, bb_grad=True)
            # s_output      : (N, 100)   main cls on normal images
            # s_rot_logit2  : (4N, 4)    rotation logits on rotated images
            # s_rot_out     : (4N, 100)  main cls on rotated images (for LT)

            with torch.no_grad():
                t_output,  _,            _ = t_model(x_nor)
                t_rot_out, t_rot_logit2, _ = t_model(x_rot)

            # ── loss1: cross-entropy on normal images ──
            loss1 = F.cross_entropy(s_output, target)

            # ── loss2: KD on normal images (Lkd) ──
            log_s_norm = F.log_softmax(s_output / args.kd_T, dim=1)
            t_norm_soft = F.softmax(t_output  / args.kd_T, dim=1).detach()
            loss2 = F.kl_div(log_s_norm, t_norm_soft,
                             reduction='batchmean') * args.kd_T ** 2

            # ── loss3: KD on transformed images (LT) — same as original ──
            aug_knowledge = F.softmax(t_rot_out / args.tf_T, dim=1).detach()
            log_aug_output = F.log_softmax(s_rot_out / args.tf_T, dim=1)

            # error-level ranking for LT (kept, same logic as original)
            aug_target_cls = target.unsqueeze(1).expand(-1, 4).reshape(-1)
            rank = torch.argsort(aug_knowledge, dim=1, descending=True)
            rank = torch.argmax(torch.eq(rank, aug_target_cls.unsqueeze(1)).long(), dim=1)
            index = torch.argsort(rank)
            tmp = torch.nonzero(rank, as_tuple=True)[0]
            wrong_num   = tmp.numel()
            correct_num = 4 * batch - wrong_num
            wrong_keep  = int(wrong_num * args.ratio_tf)
            index = index[:correct_num + wrong_keep]
            distill_index_tf = torch.sort(index)[0]

            loss3 = F.kl_div(log_aug_output[distill_index_tf],
                             aug_knowledge[distill_index_tf],
                             reduction='batchmean') * args.tf_T ** 2

            # ── loss4: rotation SS distillation (Lss) ──
            # Simply match student and teacher rotation logits via KL divergence.
            # No error-level ranking needed (rotation is a simple 4-class task).
            t_rot_soft = F.softmax(t_rot_logit2 / args.ss_T, dim=1).detach()
            log_s_rot  = F.log_softmax(s_rot_logit2 / args.ss_T, dim=1)
            loss4 = F.kl_div(log_s_rot, t_rot_soft,
                             reduction='batchmean') * args.ss_T ** 2

            loss = (args.ce_weight * loss1
                  + args.kd_weight * loss2
                  + args.tf_weight * loss3
                  + args.ss_weight * loss4)

            loss.backward()
            optimizer.step()

            cls_batch_acc = accuracy(s_output,     target,     topk=(1,))[0]
            ssp_batch_acc = accuracy(s_rot_logit2, rot_labels, topk=(1,))[0]
            loss1_record.update(loss1.item(), batch)
            loss2_record.update(loss2.item(), batch)
            loss3_record.update(loss3.item(), len(distill_index_tf))
            loss4_record.update(loss4.item(), 4 * batch)
            cls_acc_record.update(cls_batch_acc.item(), batch)
            ssp_acc_record.update(ssp_batch_acc.item(), 4 * batch)

        logger.add_scalar('train/ce_loss',  loss1_record.avg, epoch + 1)
        logger.add_scalar('train/kd_loss',  loss2_record.avg, epoch + 1)
        logger.add_scalar('train/tf_loss',  loss3_record.avg, epoch + 1)
        logger.add_scalar('train/ss_loss',  loss4_record.avg, epoch + 1)
        logger.add_scalar('train/cls_acc',  cls_acc_record.avg, epoch + 1)
        logger.add_scalar('train/ss_acc',   ssp_acc_record.avg, epoch + 1)

        run_time = time.time() - start
        print('student_train_Epoch:{:03d}/{:03d}\t run_time:{:.3f}\t '
              'ce_loss:{:.3f}\t kd_loss:{:.3f}\t cls_acc:{:.2f}'.format(
              epoch + 1, args.epoch, run_time,
              loss1_record.avg, loss2_record.avg, cls_acc_record.avg))

        if (epoch + 1) % args.val_interval == 0:
            s_model.eval()
            acc_record  = AverageMeter()
            loss_record = AverageMeter()
            start = time.time()
            for x, target in tqdm(val_loader):
                x, target = x[:,0,:,:,:].cuda(), target.cuda()
                with torch.no_grad():
                    output, _, _ = s_model(x)
                    loss = F.cross_entropy(output, target)
                batch_acc = accuracy(output, target, topk=(1,))[0]
                acc_record.update(batch_acc.item(), x.size(0))
                loss_record.update(loss.item(), x.size(0))

            run_time = time.time() - start
            logger.add_scalar('val/ce_loss',  loss_record.avg, epoch + 1)
            logger.add_scalar('val/cls_acc',  acc_record.avg,  epoch + 1)
            print('student_test_Epoch:{:03d}/{:03d}\t run_time:{:.2f}\t '
                  'cls_acc:{:.2f}\n'.format(
                  epoch + 1, args.epoch, run_time, acc_record.avg))

            if acc_record.avg > best_acc:
                best_acc = acc_record.avg
                state_dict = dict(epoch=epoch + 1,
                                  state_dict=s_model.state_dict(),
                                  best_acc=best_acc)
                name = osp.join(exp_path, 'ckpt/student_best.pth')
                os.makedirs(osp.dirname(name), exist_ok=True)
                torch.save(state_dict, name)

        scheduler.step()