import pickle
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt

USE_CUDA = True
FILE_PATH = 'E:\CV_Class\Trajectory_Prediction_Lin\checkpoints_batch2\experiment_10'


def read_pickle():
    pickle_name = os.path.join(FILE_PATH, 'parameters.txt')
    with open(pickle_name, 'rb') as f:
        content = pickle.load(f)
        print(content)


def read_pth():
    pth_name = os.path.join(FILE_PATH, 'best_model.pth')
    checkpoint = torch.load(pth_name, map_location='cpu')
    best_loss = checkpoint['best_loss']
    print(best_loss)


def loss_curve():
    loss_file = os.path.join(FILE_PATH, 'train_loss.csv')
    loss_data = pd.read_csv(loss_file)
    l2_loss, ade, fde = loss_data.iloc[:, 1], loss_data.iloc[:, 2], loss_data.iloc[:, 3]
    train_epoch = range(1, 400)
    plt.plot(train_epoch, l2_loss)
    plt.xlabel('number of epoch')
    plt.ylabel('train loss')
    plt.annotate('('+str(train_epoch[-1]+1)+', '+str(l2_loss.iloc[-1])+')', xy=(train_epoch[-1], l2_loss.iloc[-1]),
                 xytext=(240, 4.5), arrowprops=dict(facecolor='red', shrink=0.005))
    plt.show()


if __name__ == '__main__':
    # read_pth()
    # read_pickle()
    loss_curve()
