import torch
import numpy as np


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


def adjust_learning_rate(optimizers, init_lr, epoch):
    lr = init_lr * (0.5 ** (epoch // 30))

    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def get_mean_error(ret_nodes, nodes, assumedNodesPresent, trueNodesPresent):
    '''
    Parameters
    ==========
    ret_nodes : A tensor of shape pred_length x numNodes x 2
    Contains the predicted positions for the nodes
    nodes : A tensor of shape pred_length x numNodes x 2
    Contains the true positions for the nodes
    nodesPresent : A list of lists, of size pred_length
    Each list contains the nodeIDs of the nodes present at that time-step
    Returns
    =======
    Error : Mean euclidean distance between predicted trajectory and the true trajectory
    '''
    pred_length = ret_nodes.size()[0]
    error = torch.zeros(pred_length).cuda()
    counter = 0

    for tstep in range(pred_length):

        for nodeID in assumedNodesPresent:

            if nodeID not in trueNodesPresent[tstep]:
                continue

            pred_pos = ret_nodes[tstep, nodeID, :]
            true_pos = nodes[tstep, nodeID, :]

            error[tstep] += torch.norm(pred_pos - true_pos, p=2)
            counter += 1

        if counter != 0:
            error[tstep] = error[tstep] / counter

    return torch.mean(error)


def get_final_error(ret_nodes, nodes, assumedNodesPresent, trueNodesPresent):
    '''
    Parameters
    ==========
    ret_nodes : A tensor of shape pred_length x numNodes x 2
    Contains the predicted positions for the nodes
    nodes : A tensor of shape pred_length x numNodes x 2
    Contains the true positions for the nodes
    nodesPresent : A list of lists, of size pred_length
    Each list contains the nodeIDs of the nodes present at that time-step
    Returns
    =======
    Error : Mean final euclidean distance between predicted trajectory and the true trajectory
    '''
    pred_length = ret_nodes.size()[0]
    error = 0
    counter = 0

    # Last time-step
    tstep = pred_length - 1
    for nodeID in assumedNodesPresent:

        if nodeID not in trueNodesPresent[tstep]:
            continue

        pred_pos = ret_nodes[tstep, nodeID, :]
        true_pos = nodes[tstep, nodeID, :]

        error += torch.norm(pred_pos - true_pos, p=2)
        counter += 1

    if counter != 0:
        error = error / counter

    return error

