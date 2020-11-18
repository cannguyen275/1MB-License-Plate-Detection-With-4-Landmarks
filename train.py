import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
from datasets import labelFpsDataLoader, detection_collate, preproc
from utils.utils import init_log, AverageMeter, draw_output, get_state_dict
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
from models.basemodel import BaseModel
from config import cfg_plate
# from eval import evaluate

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--network', default='CCPD', type=str,
                    help='nothing')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning_rate', default=0.00001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--resume', default="weights/CCPD/CCPD_109.pth", type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--num_classes', default=2, type=int,
                    help='Number of class used in model')
parser.add_argument('--save_folder', default='./weights/', type=str,
                    help='Directory for saving checkpoint models')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--grad_accumulation_steps', default=1, type=int,
                    help='GPU id to use.')
parser.add_argument('--print_freq', default=100, type=int,
                    help='frequent of showing training status')
parser.add_argument('--eval_freq', default=2, type=int,
                    help='frequent of showing training status')


def train(train_loader, model, priors, criterion, optimizer, scheduler, epoch, logger, args):
    # switch to train mode

    model.train()
    losses = AverageMeter()
    for idx, (images, targets) in enumerate(train_loader):
        images = images.cuda()
        targets = [annot.cuda() for annot in targets]
        # print('images: ', images.size())
        # compute output
        output = model(images)
        # print('priors: ', priors.size())
        loss_l, loss_c, loss_landm = criterion(output, priors, targets)
        loss = loss_l + loss_c + loss_landm

        optimizer.zero_grad()
        # compute gradient and do SGD step
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        losses.update(loss.item())

        # Update learning rate
        # scheduler.step()

        # Print status
        if idx % args.print_freq == 0:
            draw_output([output[0][0, :, :], output[1][0, :, :], output[2][0, :, :]], images[0, :, :], cfg_plate,
                        targets[0])
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Loc: {loss_l:.3f} \t'
                        'Cla: {loss_c:.3f} \t'
                        'Landm: {loss_lm:.3f} \t'
                        'Learning rate: {learning_rate:.5f}'.format(epoch, idx, len(train_loader),
                                                                    loss=losses,
                                                                    loss_l=loss_l.item(),
                                                                    loss_c=loss_c,
                                                                    loss_lm=loss_landm,
                                                                    learning_rate=optimizer.param_groups[0]['lr']))
    return losses.avg


def main_worker(args):
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    # Log in Tensorboard
    writer = SummaryWriter()
    # log init
    save_dir = os.path.join('logs',
                            'train' + '_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    if os.path.exists(save_dir):
        raise NameError('model dir exists!')
    os.makedirs(save_dir)
    logger = init_log(save_dir)

    train_dataset = labelFpsDataLoader("/home/can/AI_Camera/Dataset/License_Plate/CCPD2019/ccpd_base",
                                       preproc=preproc(cfg_plate['image_size'], (104, 117, 123)))
    # valid_dataset = ValDataset(os.path.join("./data/widerface/val", "data/train/label.txt"))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, collate_fn=detection_collate, pin_memory=True)
    # valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False,
    #                                            num_workers=args.workers, collate_fn=detection_collate, pin_memory=True)

    # Initialize model
    model = BaseModel(cfg=cfg_plate)

    checkpoint = []
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
        params = checkpoint['parser']
        # args = params
        args.start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['state_dict'])
        del params
        del checkpoint

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = model.cuda()
        print('Run with DataParallel ....')
        model = torch.nn.DataParallel(model).cuda()

    priorbox = PriorBox(cfg_plate)

    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.cuda()

    criterion = MultiBoxLoss(args.num_classes, 0.35, True, 0, True, 7, 0.35, False)
    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Define learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    logger.info('Step per opoch: {}'.format(len(train_loader)))

    # Start training per epoch
    recall, precision = 0, 0
    for epoch in range(args.start_epoch, args.epochs):
        train_loss = train(train_loader, model, priors, criterion, optimizer, scheduler, epoch, logger, args)

        # if epoch % args.eval_freq == 0:
        #     recall, precision = evaluate(valid_loader, model)
        #
        # logger.info('Recall: {:.4f} \t'
        #             'Prcision: {:.3f} \t'.format(recall, precision))

        # Log to Tensorboard
        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('model/train_loss', train_loss, epoch)
        writer.add_scalar('model/learning_rate', lr, epoch)
        # writer.add_scalar('model/precision', precision, epoch)
        # writer.add_scalar('model/recall', recall, epoch)

        # scheduler.step()
        scheduler.step(train_loss)
        state = {
            'epoch': epoch,
            'parser': args,
            'state_dict': get_state_dict(model)
        }
        torch.save(
            state,
            os.path.join(
                args.save_folder,
                args.network,
                "{}_{}.pth".format(args.network, epoch)))


if __name__ == '__main__':
    args = parser.parse_args()
    main_worker(args)
