import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import random

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataloader.load_data import load_GC
from models.prediction_model import build_model
from tools.keeper import Keeper
from tools.helper import AverageMeter, adjust_learning_rate


DATA_PATH = 'datasets/GC.npz'
# DATA_PATH = 'datasets/GC_8_12.npz'


def create_args():
    parser = argparse.ArgumentParser(description='Pedestrian Prediction')
    parser.add_argument('--pedestrian_num', type=int, default=20,
                        help='pedestrian number of each sample')
    parser.add_argument('--hidden_size', type=int, default=128,
                        help='size of hidden state in LSTM')

    parser.add_argument('--input_frame', type=int, default=5,
                        help='the number of observe frames')
    parser.add_argument('--input_size', type=int, default=2,
                        help='input size, here 2 means (x, y)')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='layers of LSTM')
    parser.add_argument('--bidirectional', type=int, default=1,
                        help='whether using bidirectional LSTM, 1 means does not use, 2 means use')
    parser.add_argument('--teacher', type=bool, default=False,
                        help='whether using teacher forcing in decoder net of ATT_LSTM')
    parser.add_argument('--lr_optim', type=str, default='adam',
                        choices=['adam', 'amsgrad', 'adam_separate', 'sgd'],
                        help='optimizer using for the model')

    parser.add_argument('--target_frame', type=int, default=5,
                        help='number of target frames')
    parser.add_argument('--target_size', type=int, default=2,
                        help='output size, here 2 means (x, y)')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size for training')
    parser.add_argument('--lr', type=float, default=3e-3,
                        help='learning rate')
    parser.add_argument('--final_lr', default=0.00001, help=
                        'final learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-3,
                        help='w-decay (default: 5e-4)')
    parser.add_argument('--val_interval', type=int, default=5,
                        help='test step')
    parser.add_argument('--epochs', type=int, default=400,
                        help='number of epoch')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='the start epoch')
    parser.add_argument('--dropout', type=float, default=0,
                        help='dropout of lstm')

    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')

    # cuda
    parser.add_argument('--use_cuda', type=bool, default=False)
    parser.add_argument('--gpu_ids', type=str, default=[0],
                        help='use which gpu to train, must be a \
                            comma-separated list of integers only (default=0)')

    parser.add_argument('--dataset', type=str, default=DATA_PATH)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--backbone', type=str, default='ATT_LSTM_Spatial',
                        choices=['CIDNN', 'SR_LSTM', 'ATT_LSTM', 'ATT_LSTM_Spatial'])
    parser.add_argument('--att_method', type=str, default='concat',
                        help='attention method', choices=['dot', 'general', 'concat'])
    return parser


def get_loss(pred, target, pedestrian_num=20):
    L2_square_loss = ((target - pred) ** 2).sum() / pedestrian_num
    ADE_loss = ((target - pred) ** 2).sum(3).sqrt().mean()
    FDE_loss = ((target - pred) ** 2).sum(3).sqrt()[:, :, -1].mean()

    return L2_square_loss, ADE_loss, FDE_loss


def validation(args, model, epoch, val_loader):
    model.eval()
    log.info('Validation: epoch [%d/%d] \n' % (epoch, args.epochs))

    ADE_loss_meter = AverageMeter()
    FDE_loss_meter = AverageMeter()
    L2_square_loss_meter = AverageMeter()

    for i, (val_input_traces, val_target_traces) in enumerate(val_loader):
        if args.use_cuda:
            val_input_traces = val_input_traces.cuda()
            val_target_traces = val_target_traces.cuda()

        with torch.no_grad():
            if args.teacher:
                regression_traces = model(val_input_traces, val_target_traces, val_mode=True)
            else:
                regression_traces = model(val_input_traces)

        if i % 2 == 1:
            keeper.save_img(val_input_traces[1, 1, :, :], val_target_traces[1, 1, :, :],
                            regression_traces[1, 1, :, :], 'epoch_{}_val_{}.jpg'.format(epoch, i))

        L2_square_loss, ADE_loss, FDE_loss = get_loss(regression_traces, val_target_traces)
        ADE_loss_meter.update(ADE_loss.item())
        FDE_loss_meter.update(FDE_loss.item())
        L2_square_loss_meter.update(L2_square_loss.item())

    val_loss = L2_square_loss_meter.avg, ADE_loss_meter.avg, FDE_loss_meter.avg

    log.info('Validation Epoch: [%d/%d], L2_suqare_loss: %.9f, ADE_loss: %.9f, FDE_loss: %.9f \n' %
             (epoch, args.epochs, val_loss[0], val_loss[1], val_loss[2]))

    return val_loss


def main(args):
    train_loader, val_loader = load_GC(args)

    log.info('building model ... \n')

    model = build_model(args)
    model = model.cuda() if args.use_cuda else model

    if args.lr_optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.lr_optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        lr_schedule = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5,
                                        verbose=True, threshold=0.0001, threshold_mode='rel',
                                        cooldown=3, min_lr=args.final_lr, eps=1e-08)
    elif args.lr_optim == 'amsgrad':
        lr_schedule = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
    elif args.lr_optim == 'adam_separate':
        encoder_optimizer = optim.Adam(model.encoder_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        decoder_optimizer = optim.Adam(model.decoder_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        regression_optimizer = optim.Adam(model.regression_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Whether using checkpoint
    if args.resume:
        check_path = os.path.join('checkpoints', args.resume)
        if not os.path.exists(check_path):
            raise RuntimeError("=> no checkpoint found")
        checkpoint = torch.load(check_path)
        if args.use_cuda:
            model.module.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])

        if args.lr_optim == 'adam_separate':
            encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
            decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
            regression_optimizer.load_state_dict(checkpoint['regression_optimizer'])
        else:
            optimizer.load_state_dict(checkpoint['optimizer'])
        best_loss = checkpoint['best_loss']
        args.start_epoch = checkpoint['epoch'] + 1
    else:
        best_loss = np.inf

    for epoch in range(args.start_epoch, args.epochs):
        s_time = time.time()
        # ------------------- Training ----------------------
        model.train()
        log.info('training: epoch [%d/%d] \n' % (epoch, args.epochs))

        ADE_loss_meter = AverageMeter()
        FDE_loss_meter = AverageMeter()
        L2_square_loss_meter = AverageMeter()
        # adjust_learning_rate([encoder_optimizer, decoder_optimizer, regression_optimizer], args.lr, epoch)
        # adjust_learning_rate([optimizer], args.lr, epoch)

        for i, (train_input_traces, train_target_traces) in enumerate(train_loader):
            if args.use_cuda:
                train_input_traces = train_input_traces.cuda()
                train_target_traces = train_target_traces.cuda()

            if args.lr_optim == 'adam_separate':
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                regression_optimizer.zero_grad()
            else:
                optimizer.zero_grad()

            if args.teacher:
                regression_traces = model(train_input_traces, train_target_traces)
            else:
                regression_traces = model(train_input_traces)

            L2_square_loss, ADE_loss, FDE_loss = get_loss(regression_traces, train_target_traces)
            ADE_loss_meter.update(ADE_loss.item())
            FDE_loss_meter.update(FDE_loss.item())
            L2_square_loss_meter.update(L2_square_loss.item())

            L2_square_loss.backward()
            if args.lr_optim == 'adam_separate':
                encoder_optimizer.step()
                decoder_optimizer.step()
                regression_optimizer.step()
            else:
                optimizer.step()

        train_loss = L2_square_loss_meter.avg, ADE_loss_meter.avg, FDE_loss_meter.avg
        keeper.save_loss(train_loss)

        if args.lr_optim == 'sgd':
            lr_schedule.step(train_loss[0])

        log.info('Train Epoch: [%d/%d], L2_suqare_loss: %.9f, ADE_loss: %.9f, FDE_loss: %.9f \n' %
                 (epoch, args.epochs, train_loss[0], train_loss[1], train_loss[2]))

        # ----------------------- Validation --------------------
        if epoch % args.val_interval == 0:
            val_loss = validation(args, model, epoch, val_loader)
            keeper.save_loss(val_loss, file_name='validation_loss.csv')

            if val_loss[1] < best_loss:
                best_loss = val_loss[1]
                if args.lr_optim == 'adam_separate':
                    keeper.save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'encoder_optimizer': encoder_optimizer.state_dict(),
                        'decoder_optimizer': decoder_optimizer.state_dict(),
                        'regression_optimizer': regression_optimizer.state_dict(),
                        'best_loss': best_loss,
                    }, 'best_model.pth')

                else:
                    keeper.save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_loss': best_loss,
                    }, 'best_model.pth')

        if args.lr_optim == 'adam_separate':
            keeper.save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'encoder_optimizer': encoder_optimizer.state_dict(),
                'decoder_optimizer': decoder_optimizer.state_dict(),
                'regression_optimizer': regression_optimizer.state_dict(),
                # 'optimizer': optimizer.state_dict(),
                'best_loss': best_loss,
            })
        else:
            keeper.save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_loss': best_loss,
            })

        e_time = time.time()
        log.info('The {0}th epoch time is {1}\n'.format(epoch, e_time - s_time))


def set_random_seed(random_seed=0):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    random.seed(random_seed)


if __name__ == '__main__':
    set_random_seed()
    parser = create_args()
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in args.gpu_ids)
    args.use_cuda = torch.cuda.is_available()

    keeper = Keeper(args)
    keeper.save_experiment_config()
    log = keeper.setup_logger()

    log.info("Using the same optimizer for the model, and optimizer is Adam with amsgrad=False.\n"
             "The backward loss is L2_loss \n"
             " The input frame is 5, and predict frame is 5 \n")

    start = time.time()
    main(args)
    end = time.time()
    log.info('Total train time: {}'.format(end - start))



