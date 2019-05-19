import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
from torch import optim

from dataloader.load_data import load_GC
from models.prediction_model import build_model
from tools.keeper import Keeper
from tools.helper import AverageMeter, adjust_learning_rate


DATA_PATH = 'datasets/GC.npz'


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

    parser.add_argument('--target_frame', type=int, default=5,
                        help='number of target frames')
    parser.add_argument('--target_size', type=int, default=2,
                        help='output size, here 2 means (x, y)')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size for training')
    parser.add_argument('--lr', type=float, default=2e-3,
                        help='learning rate')
    parser.add_argument('--final_lr', default=0.0001, help=
                        'final learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-3,
                        help='w-decay (default: 5e-4)')
    parser.add_argument('--val_interval', type=int, default=5,
                        help='test step')
    parser.add_argument('--epochs', type=int, default=10000,
                        help='number of epoch')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='the start epoch')

    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')

    # cuda
    parser.add_argument('--use_cuda', type=bool, default=False)
    parser.add_argument('--gpu_ids', type=str, default=[0],
                        help='use which gpu to train, must be a \
                            comma-separated list of integers only (default=0)')

    parser.add_argument('--dataset', type=str, default=DATA_PATH)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--backbone', type=str, default='CIDNN',
                        choices=['CIDNN', 'SR_LSTM'])
    return parser


def get_loss(pred, target, pedestrian_num=20):
    L2_square_loss = ((target - pred) ** 2).sum() / pedestrian_num
    MSE_loss = ((target - pred) ** 2).sum(3).sqrt().mean()

    return L2_square_loss, MSE_loss


def validation(args, model, epoch, val_loader):
    model.eval()
    log.info('Validation: epoch [%d/%d] \n' % (epoch, args.epochs))

    MSE_loss_meter = AverageMeter()
    L2_square_loss_meter = AverageMeter()

    for i, (val_input_traces, val_target_traces) in enumerate(val_loader):
        if args.use_cuda:
            val_input_traces = val_input_traces.cuda()
            val_target_traces = val_target_traces.cuda()

        with torch.no_grad():
            regression_traces = model(val_input_traces)

        if i % 9 == 1:
            keeper.save_img(val_input_traces[1, 1, :, :], val_target_traces[1, 1, :, :],
                            regression_traces[1, 1, :, :], 'epoch_{}_val_{}.jpg'.format(epoch, i))

        L2_square_loss, MSE_loss = get_loss(regression_traces, val_target_traces)
        MSE_loss_meter.update(MSE_loss.item())
        L2_square_loss_meter.update(L2_square_loss.item())

    val_MSE_loss, val_L2_loss = MSE_loss_meter.avg, L2_square_loss_meter.avg

    log.info('Epoch: [%d/%d], L2_suqare_loss: %.9f, MSE_loss: %.9f' %
             (epoch, args.epochs, val_L2_loss, val_MSE_loss))

    return val_MSE_loss


def main(args):
    train_loader, val_loader = load_GC(args)

    log.info('building model ... \n')

    model = build_model(args)
    model = torch.nn.DataParallel(model).cuda() if args.use_cuda else model

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
        encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
        decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])
        regression_optimizer.load_state_dict(checkpoint['regression_optimizer'])
        best_loss = checkpoint['best_loss']
        args.start_epoch = checkpoint['epoch'] + 1
    else:
        best_loss = np.inf

    for epoch in range(args.start_epoch, args.epochs):
        s_time = time.time()
        # ------------------- Training ----------------------
        model.train()
        log.info('training: epoch [%d/%d] \n' % (epoch, args.epochs))

        MSE_loss_meter = AverageMeter()
        L2_square_loss_meter = AverageMeter()
        adjust_learning_rate([encoder_optimizer, decoder_optimizer, regression_optimizer], args.lr, epoch)

        for i, (train_input_traces, train_target_traces) in enumerate(train_loader):
            if args.use_cuda:
                train_input_traces = train_input_traces.cuda()
                train_target_traces = train_target_traces.cuda()

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            regression_optimizer.zero_grad()

            regression_traces = model(train_input_traces)

            L2_square_loss, MSE_loss = get_loss(regression_traces, train_target_traces)
            MSE_loss_meter.update(MSE_loss.item())
            L2_square_loss_meter.update(L2_square_loss.item())

            L2_square_loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            regression_optimizer.step()

        train_MSE_loss, train_L2_loss = MSE_loss_meter.avg, L2_square_loss_meter.avg

        log.info('Epoch: [%d/%d], L2_suqare_loss: %.9f, MSE_loss: %.9f \n' %
                 (epoch, args.epochs, train_L2_loss, train_MSE_loss))

        # ----------------------- Validation --------------------
        if epoch % args.val_interval == 0:
            val_MSE_loss = validation(args, model, epoch, val_loader)

            if val_MSE_loss < best_loss:
                best_loss = val_MSE_loss
                keeper.save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.module.state_dict() if args.use_cuda else model.state_dict(),
                    'encoder_optimizer': encoder_optimizer.state_dict(),
                    'decoder_optimizer': decoder_optimizer.state_dict(),
                    'regression_optimizer': regression_optimizer.state_dict(),
                    'best_loss': best_loss,
                }, 'best_model.pth')

        keeper.save_checkpoint({
            'epoch': epoch,
            'state_dict': model.module.state_dict() if args.use_cuda else model.state_dict(),
            'encoder_optimizer': encoder_optimizer.state_dict(),
            'decoder_optimizer': decoder_optimizer.state_dict(),
            'regression_optimizer': regression_optimizer.state_dict(),
            'best_loss': best_loss,
        })

        e_time = time.time()
        log.info('one epoch time is {}'.format(e_time - s_time))


if __name__ == '__main__':
    parser = create_args()
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in args.gpu_ids)
    args.use_cuda = torch.cuda.is_available()

    keeper = Keeper(args)
    keeper.save_experiment_config()
    log = keeper.setup_logger()

    start = time.time()
    main(args)
    end = time.time()
    log.info('Total train time: {}'.format(end - start))



