import torch.nn as nn

from model.prediction_model import CIDNN_Model, SRLSTM_Model


def build_model(args):
    if args.backbone == 'CIDNN':
        return CIDNN_Model(args)
    elif args.backbone == 'SR_LSTM':
        return SRLSTM_Model(args)


def initial_fc_weights(modules):
    for m in modules:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

