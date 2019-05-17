import numpy as np
import torch


def load_GC(args):
    # load data
    data = np.load(args.dataset)
    train_X, train_Y = data['train_X'], data['train_Y']
    val_X, val_Y = data['test_X'], data['test_Y']

    if args.batch_size <= 0:
        args.batch_size = train_X.shape[0]

    val_input_traces = torch.FloatTensor(val_X)
    val_target_traces = torch.FloatTensor(val_Y)
    val = torch.utils.data.TensorDataset(val_input_traces, val_target_traces)
    val_loader = torch.utils.data.DataLoader(val, batch_size=args.batch_size,
                                             shuffle=True, num_workers=4)

    # (B, pedestrian_num, frame_size, 2)
    train_input_traces = torch.FloatTensor(train_X)
    # (B, pedestrian_num, frame_size, 2)
    train_target_traces = torch.FloatTensor(train_Y)

    train_input_traces = train_input_traces
    train_target_traces = train_target_traces

    # data loader
    train = torch.utils.data.TensorDataset(train_input_traces, train_target_traces)
    train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size,
                                               shuffle=True, num_workers=4)
    return train_loader, val_loader
