import torch
import numpy as np


def Gaussian2DLikelihood(outputs, targets, nodesPresent, pred_length):
    '''
    Parameters:
    outputs: Torch variable containing tensor of shape seq_length x numNodes x 1 x output_size
    targets: Torch variable containing tensor of shape seq_length x numNodes x 1 x input_size
    nodesPresent : A list of lists, of size seq_length. Each list contains the nodeIDs that are present in the frame
    '''
    seq_length = outputs.size()[0]
    obs_length = seq_length - pred_length

    # Extract mean, std devs and correlation
    mux, muy, sx, sy, corr = getCoef(outputs)

    # Compute factors
    normx = targets[:, :, 0] - mux
    normy = targets[:, :, 1] - muy
    sxsy = sx * sy

    z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
    negRho = 1 - corr**2

    # Numerator
    result = torch.exp(-z/(2*negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon))

    loss = 0
    counter = 0

    for framenum in range(obs_length, seq_length):
        nodeIDs = nodesPresent[framenum]

        for nodeID in nodeIDs:

            loss = loss + result[framenum, nodeID]
            counter = counter + 1

    if counter != 0:
        return loss / counter
    else:
        return loss
