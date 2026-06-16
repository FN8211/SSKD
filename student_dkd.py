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
from wrapper import wrapper
from cifar import CIFAR100
from tqdm import tqdm

from models import model_dict

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='train SSKD student network.')
parser.add_argument('--epoch', type=int, default=240)
parser.add_argument('--t-epoch', type=int, default=60)
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--val-interval', type=int, default=1)

parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--t-lr', type=float, default=0.05)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--milestones', type=int, nargs='+', default=[150,180,210])
parser.add_argument('--t-milestones', type=int, nargs='+', default=[30,45])

parser.add_argument('--save-interval', type=int, default=40)
parser.add_argument('--ce-weight', type=float, default=0.1)
parser.add_argument('--kd-weight', type=float, default=0.9)
parser.add_argument('--tf-weight', type=float, default=2.7)
parser.add_argument('--ss-weight', type=float, default=10.0)
parser.add_argument('--tckd-weight', type=float, default=1.0)
parser.add_argument('--nckd-weight', type=float, default=8.0)

parser.add_argument('--kd-T', type=float, default=4.0)
parser.add_argument('--tf-T', type=float, default=4.0)
parser.add_argument('--ss-T', type=float, default=0.5)

parser.add_argument('--ratio-tf', type=float, default=1.0)
parser.add_argument('--ratio-ss', type=float, default=0.75)
parser.add_argument('--s-arch', type=str)
parser.add_argument('--t-path', type=str)
parser.add_argument('--t-ckpt', type=str, default=None)

parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--gpu-id', type=int, default=0)

args = parser.parse_args()

if __name__ == '__main__':
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    t_name = os.path.basename(os.path.normpath(args.t_path))
    t_arch = '_'.join(t_name.split('_')[1:])
    exp_name = 'sskd_dkd_student_{}_weight{}+{}+{}+{}_tckd{}+nckd{}_T{}+{}+{}_ratio{}+{}_seed{}_{}'.format(
                args.s_arch,
                args.ce_weight, args.kd_weight, args.tf_weight, args.ss_weight,
                args.tckd_weight, args.nckd_weight,
                args.kd_T, args.tf_T, args.ss_T,
                args.ratio_tf, args.ratio_ss,
                args.seed, t_name)
    exp_path = './experiments/{}'.format(exp_name)
    os.makedirs(exp_path, exist_ok=True)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2675, 0.2565, 0.2761]),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4866, 0.4409], std=[0.2675, 0.2565, 0.2761]),
    ])

    trainset = CIFAR100('./data', train=True, transform=transform_train)
    valset = CIFAR100('./data', train=False, transform=transform_test)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=False)

    ckpt_path = osp.join(args.t_path, 'ckpt/best.pth')
    t_model = model_dict[t_arch](num_classes=100).cuda()
    state_dict = torch.load(ckpt_path)['state_dict']
    t_model.load_state_dict(state_dict)
    t_model = wrapper(module=t_model).cuda()

    t_optimizer = optim.SGD([{'params':t_model.backbone.parameters(), 'lr':0.0},
                            {'params':t_model.proj_head.parameters(), 'lr':args.t_lr}],
                            momentum=args.momentum, weight_decay=args.weight_decay)
    t_model.eval()
    t_scheduler = MultiStepLR(t_optimizer, milestones=args.t_milestones, gamma=args.gamma)

    logger = SummaryWriter(osp.join(exp_path, 'events'))

    acc_record = AverageMeter()
    loss_record = AverageMeter()
    start = time.time()
    for x, target in tqdm(val_loader):
        x = x[:,0,:,:,:].cuda()
        target = target.cuda()
        with torch.no_grad():
            output, _, feat = t_model(x)
            loss = F.cross_entropy(output, target)
        batch_acc = accuracy(output, target, topk=(1,))[0]
        acc_record.update(batch_acc.item(), x.size(0))
        loss_record.update(loss.item(), x.size(0))

    run_time = time.time() - start
    info = 'teacher cls_acc:{:.2f}\n'.format(acc_record.avg)
    print(info)

    # train ssp_head 或直接加载
    if args.t_ckpt is not None:
        t_model.load_state_dict(torch.load(args.t_ckpt))
        print('loaded teacher ckpt from {}'.format(args.t_ckpt))
    else:
        for epoch in range(args.t_epoch):

            t_model.eval()
            loss_record = AverageMeter()
            acc_record = AverageMeter()

            start = time.time()
            for x, _ in tqdm(train_loader):

                t_optimizer.zero_grad()

                x = x.cuda()
                c,h,w = x.size()[-3:]
                x = x.view(-1, c, h, w)

                _, rep, feat = t_model(x, bb_grad=False)
                batch = int(x.size(0) / 4)
                nor_index = (torch.arange(4*batch) % 4 == 0).cuda()
                aug_index = (torch.arange(4*batch) % 4 != 0).cuda()

                nor_rep = rep[nor_index]
                aug_rep = rep[aug_index]
                nor_rep = nor_rep.unsqueeze(2).expand(-1,-1,3*batch).transpose(0,2)
                aug_rep = aug_rep.unsqueeze(2).expand(-1,-1,1*batch)
                simi = F.cosine_similarity(aug_rep, nor_rep, dim=1)
                target = torch.arange(batch).unsqueeze(1).expand(-1,3).contiguous().view(-1).long().cuda()
                loss = F.cross_entropy(simi, target)

                loss.backward()
                t_optimizer.step()

                batch_acc = accuracy(simi, target, topk=(1,))[0]
                loss_record.update(loss.item(), 3*batch)
                acc_record.update(batch_acc.item(), 3*batch)

            logger.add_scalar('train/teacher_ssp_loss', loss_record.avg, epoch+1)
            logger.add_scalar('train/teacher_ssp_acc', acc_record.avg, epoch+1)

            run_time = time.time() - start
            info = 'teacher_train_Epoch:{:03d}/{:03d}\t run_time:{:.3f}\t ssp_loss:{:.3f}\t ssp_acc:{:.2f}\t'.format(
                epoch+1, args.t_epoch, run_time, loss_record.avg, acc_record.avg)
            print(info)

            if (epoch + 1) % args.val_interval == 0:
                t_model.eval()
                acc_record = AverageMeter()
                loss_record = AverageMeter()
                start = time.time()
                for x, _ in tqdm(val_loader):
                    x = x.cuda()
                    c,h,w = x.size()[-3:]
                    x = x.view(-1, c, h, w)
                    with torch.no_grad():
                        _, rep, feat = t_model(x)
                    batch = int(x.size(0) / 4)
                    nor_index = (torch.arange(4*batch) % 4 == 0).cuda()
                    aug_index = (torch.arange(4*batch) % 4 != 0).cuda()
                    nor_rep = rep[nor_index]
                    aug_rep = rep[aug_index]
                    nor_rep = nor_rep.unsqueeze(2).expand(-1,-1,3*batch).transpose(0,2)
                    aug_rep = aug_rep.unsqueeze(2).expand(-1,-1,1*batch)
                    simi = F.cosine_similarity(aug_rep, nor_rep, dim=1)
                    target = torch.arange(batch).unsqueeze(1).expand(-1,3).contiguous().view(-1).long().cuda()
                    loss = F.cross_entropy(simi, target)
                    batch_acc = accuracy(simi, target, topk=(1,))[0]
                    acc_record.update(batch_acc.item(),3*batch)
                    loss_record.update(loss.item(), 3*batch)

                run_time = time.time() - start
                logger.add_scalar('val/teacher_ssp_loss', loss_record.avg, epoch+1)
                logger.add_scalar('val/teacher_ssp_acc', acc_record.avg, epoch+1)

                info = 'ssp_test_Epoch:{:03d}/{:03d}\t run_time:{:.2f}\t ssp_loss:{:.3f}\t ssp_acc:{:.2f}\n'.format(
                        epoch+1, args.t_epoch, run_time, loss_record.avg, acc_record.avg)
                print(info)

            t_scheduler.step()

        name = osp.join(exp_path, 'ckpt/teacher.pth')
        os.makedirs(osp.dirname(name), exist_ok=True)
        torch.save(t_model.state_dict(), name)

    s_model = model_dict[args.s_arch](num_classes=100)
    s_model = wrapper(module=s_model).cuda()
    optimizer = optim.SGD(s_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    best_acc = 0
    for epoch in range(args.epoch):

        s_model.train()
        loss1_record = AverageMeter()
        loss2_record = AverageMeter()
        loss2_tckd_record = AverageMeter()
        loss2_nckd_record = AverageMeter()
        loss3_record = AverageMeter()
        loss4_record = AverageMeter()
        cls_acc_record = AverageMeter()
        ssp_acc_record = AverageMeter()

        start = time.time()
        for x, target in tqdm(train_loader):

            optimizer.zero_grad()

            c,h,w = x.size()[-3:]
            x = x.view(-1,c,h,w).cuda()
            target = target.cuda()

            batch = int(x.size(0) / 4)
            nor_index = (torch.arange(4*batch) % 4 == 0).cuda()
            aug_index = (torch.arange(4*batch) % 4 != 0).cuda()

            output, s_feat, _ = s_model(x, bb_grad=True)
            log_aug_output = F.log_softmax(output[aug_index] / args.tf_T, dim=1)
            with torch.no_grad():
                knowledge, t_feat, _ = t_model(x)
                aug_knowledge = F.softmax(knowledge[aug_index] / args.tf_T, dim=1)

            # error level ranking
            aug_target = target.unsqueeze(1).expand(-1,3).contiguous().view(-1).long().cuda()
            rank = torch.argsort(aug_knowledge, dim=1, descending=True)
            rank = torch.argmax(torch.eq(rank, aug_target.unsqueeze(1)).long(), dim=1)
            index = torch.argsort(rank)
            tmp = torch.nonzero(rank, as_tuple=True)[0]
            wrong_num = tmp.numel()
            correct_num = 3*batch - wrong_num
            wrong_keep = int(wrong_num * args.ratio_tf)
            index = index[:correct_num+wrong_keep]
            distill_index_tf = torch.sort(index)[0]

            s_nor_feat = s_feat[nor_index]
            s_aug_feat = s_feat[aug_index]
            s_nor_feat = s_nor_feat.unsqueeze(2).expand(-1,-1,3*batch).transpose(0,2)
            s_aug_feat = s_aug_feat.unsqueeze(2).expand(-1,-1,1*batch)
            s_simi = F.cosine_similarity(s_aug_feat, s_nor_feat, dim=1)

            t_nor_feat = t_feat[nor_index]
            t_aug_feat = t_feat[aug_index]
            t_nor_feat = t_nor_feat.unsqueeze(2).expand(-1,-1,3*batch).transpose(0,2)
            t_aug_feat = t_aug_feat.unsqueeze(2).expand(-1,-1,1*batch)
            t_simi = F.cosine_similarity(t_aug_feat, t_nor_feat, dim=1)

            t_simi = t_simi.detach()
            aug_target = torch.arange(batch).unsqueeze(1).expand(-1,3).contiguous().view(-1).long().cuda()
            rank = torch.argsort(t_simi, dim=1, descending=True)
            rank = torch.argmax(torch.eq(rank, aug_target.unsqueeze(1)).long(), dim=1)
            index = torch.argsort(rank)
            tmp = torch.nonzero(rank, as_tuple=True)[0]
            wrong_num = tmp.numel()
            correct_num = 3*batch - wrong_num
            wrong_keep = int(wrong_num * args.ratio_ss)
            index = index[:correct_num+wrong_keep]
            distill_index_ss = torch.sort(index)[0]

            log_simi = F.log_softmax(s_simi / args.ss_T, dim=1)
            simi_knowledge = F.softmax(t_simi / args.ss_T, dim=1)

            # loss1: CE
            loss1 = F.cross_entropy(output[nor_index], target)

            # loss2: DKD
            nor_output = output[nor_index]
            nor_knowledge = knowledge[nor_index].detach()

            # 教师/学生 目标类概率
            p_t = F.softmax(nor_knowledge / args.kd_T, dim=1).gather(1, target.unsqueeze(1))  # [batch, 1]
            p_s = F.softmax(nor_output / args.kd_T, dim=1).gather(1, target.unsqueeze(1))     # [batch, 1]

            # TCKD: 目标类 vs 非目标类 二元KL
            b_t = torch.cat([p_t, 1 - p_t], dim=1)  # [batch, 2]
            b_s = torch.cat([p_s, 1 - p_s], dim=1)  # [batch, 2]
            loss_tckd = F.kl_div(torch.log(b_s + 1e-8), b_t, reduction='batchmean')

            # NCKD: 非目标类之间的分布KL（GPU友好实现）
            num_classes = nor_knowledge.size(1)
            target_mask = torch.zeros(batch, num_classes).cuda().scatter_(1, target.unsqueeze(1), 1.0)

            nor_knowledge_soft = F.softmax(nor_knowledge / args.kd_T, dim=1)
            nor_output_soft = F.softmax(nor_output / args.kd_T, dim=1)

            nor_knowledge_nontarget = nor_knowledge_soft * (1 - target_mask)
            nor_output_nontarget = nor_output_soft * (1 - target_mask)

            nor_knowledge_nontarget = nor_knowledge_nontarget / nor_knowledge_nontarget.sum(dim=1, keepdim=True)
            nor_output_nontarget = nor_output_nontarget / nor_output_nontarget.sum(dim=1, keepdim=True)

            loss_nckd = F.kl_div(torch.log(nor_output_nontarget + 1e-8), nor_knowledge_nontarget, reduction='batchmean')

            loss2 = (args.tckd_weight * loss_tckd + args.nckd_weight * loss_nckd) * args.kd_T * args.kd_T

            # loss3: TF
            loss3 = F.kl_div(log_aug_output[distill_index_tf], aug_knowledge[distill_index_tf],
                            reduction='batchmean') * args.tf_T * args.tf_T

            # loss4: SS
            loss4 = F.kl_div(log_simi[distill_index_ss], simi_knowledge[distill_index_ss],
                            reduction='batchmean') * args.ss_T * args.ss_T

            loss = args.ce_weight * loss1 \
                 + args.kd_weight * loss2 \
                 + args.tf_weight * loss3 \
                 + args.ss_weight * loss4

            loss.backward()
            optimizer.step()

            cls_batch_acc = accuracy(output[nor_index], target, topk=(1,))[0]
            ssp_batch_acc = accuracy(s_simi, aug_target, topk=(1,))[0]
            loss1_record.update(loss1.item(), batch)
            loss2_record.update(loss2.item(), batch)
            loss2_tckd_record.update(loss_tckd.item(), batch)
            loss2_nckd_record.update(loss_nckd.item(), batch)
            loss3_record.update(loss3.item(), len(distill_index_tf))
            loss4_record.update(loss4.item(), len(distill_index_ss))
            cls_acc_record.update(cls_batch_acc.item(), batch)
            ssp_acc_record.update(ssp_batch_acc.item(), 3*batch)

        logger.add_scalar('train/ce_loss', loss1_record.avg, epoch+1)
        logger.add_scalar('train/kd_loss', loss2_record.avg, epoch+1)
        logger.add_scalar('train/tckd_loss', loss2_tckd_record.avg, epoch+1)
        logger.add_scalar('train/nckd_loss', loss2_nckd_record.avg, epoch+1)
        logger.add_scalar('train/tf_loss', loss3_record.avg, epoch+1)
        logger.add_scalar('train/ss_loss', loss4_record.avg, epoch+1)
        logger.add_scalar('train/cls_acc', cls_acc_record.avg, epoch+1)
        logger.add_scalar('train/ss_acc', ssp_acc_record.avg, epoch+1)

        run_time = time.time() - start
        info = 'student_train_Epoch:{:03d}/{:03d}\t run_time:{:.3f}\t ce_loss:{:.3f}\t kd_loss:{:.3f}\t tckd:{:.3f}\t nckd:{:.3f}\t tf_loss:{:.3f}\t ss_loss:{:.3f}\t cls_acc:{:.2f}'.format(
            epoch+1, args.epoch, run_time, loss1_record.avg, loss2_record.avg,
            loss2_tckd_record.avg, loss2_nckd_record.avg, loss3_record.avg, loss4_record.avg, cls_acc_record.avg)
        print(info)

        if (epoch + 1) % args.val_interval == 0:
            s_model.eval()
            acc_record = AverageMeter()
            loss_record = AverageMeter()
            start = time.time()
            for x, target in tqdm(val_loader):
                x = x[:,0,:,:,:].cuda()
                target = target.cuda()
                with torch.no_grad():
                    output, _, feat = s_model(x)
                    loss = F.cross_entropy(output, target)
                batch_acc = accuracy(output, target, topk=(1,))[0]
                acc_record.update(batch_acc.item(), x.size(0))
                loss_record.update(loss.item(), x.size(0))

            run_time = time.time() - start
            logger.add_scalar('val/ce_loss', loss_record.avg, epoch+1)
            logger.add_scalar('val/cls_acc', acc_record.avg, epoch+1)

            info = 'student_test_Epoch:{:03d}/{:03d}\t run_time:{:.2f}\t cls_acc:{:.2f}\n'.format(
                    epoch+1, args.epoch, run_time, acc_record.avg)
            print(info)

            if acc_record.avg > best_acc:
                best_acc = acc_record.avg
                state_dict = dict(epoch=epoch+1, state_dict=s_model.state_dict(), best_acc=best_acc)
                name = osp.join(exp_path, 'ckpt/student_best.pth')
                os.makedirs(osp.dirname(name), exist_ok=True)
                torch.save(state_dict, name)

        scheduler.step()