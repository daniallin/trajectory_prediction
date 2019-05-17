'''
Visualization script for the structural RNN model
introduced in https://arxiv.org/abs/1511.05298
Author : Anirudh Vemula
Date : 3rd April 2017
'''
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse


def plot_trajectories(true_trajs, pred_trajs, nodesPresent, obs_length, name, plot_directory, withBackground=False):
    '''
    Parameters
    ==========
    true_trajs : Numpy matrix of shape seq_length x numNodes x 2
    Contains the true trajectories of the nodes
    pred_trajs : Numpy matrix of shape seq_length x numNodes x 2
    Contains the predicted trajectories of the nodes
    nodesPresent : A list of lists, of size seq_length
    Each list contains the nodeIDs present at that time-step
    obs_length : Length of observed trajectory
    name : Name of the plot
    withBackground : Include background or not
    '''

    traj_length, numNodes, _ = true_trajs.shape

    plt.figure()

    # Load the background
    im = plt.imread('plot/background.png')
    width_true = im.shape[0]
    height_true = im.shape[1]

    if withBackground:
       implot = plt.imshow(im)
       width = width_true
       height = height_true
    else:
        width = 1
        height = 1

    traj_data = {}
    for tstep in range(traj_length):
        pred_pos = pred_trajs[tstep, :]
        true_pos = true_trajs[tstep, :]

        for ped in range(numNodes):
            if ped not in traj_data and tstep < obs_length:
                traj_data[ped] = [[], []]

            if ped in nodesPresent[tstep]:
                traj_data[ped][0].append(true_pos[ped, :])
                traj_data[ped][1].append(pred_pos[ped, :])

    for j in traj_data:
        c = np.random.rand(3, 1)
        true_traj_ped = traj_data[j][0]  # List of [x,y] elements
        pred_traj_ped = traj_data[j][1]

        true_x = [(p[0]+1)/2*height for p in true_traj_ped]
        true_y = [(p[1]+1)/2*width for p in true_traj_ped]
        pred_x = [(p[0]+1)/2*height for p in pred_traj_ped]
        pred_y = [(p[1]+1)/2*width for p in pred_traj_ped]

        plt.plot(true_x, true_y, color=c, linestyle='solid', marker='o')
        plt.plot(pred_x, pred_y, color=c, linestyle='dashed', marker='x')

    if not withBackground:
        plt.ylim((1, 0))
        plt.xlim((0, 1))

    # plt.show()
    if withBackground:
        plt.savefig(plot_directory + name + '_with_bg.png')
    else:
        plt.savefig(plot_directory+'/'+name+'.png')

    plt.gcf().clear()
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    # Experiments

    parser.add_argument('--test_dataset', type=int, default=0,
                        help='test dataset index')

    # Parse the parameters
    args = parser.parse_args()

    # Save directory
    save_directory = 'save/'
    save_directory += str(args.test_dataset) + '/'
    plot_directory = 'plot/'

    f = open(save_directory+'/results.pkl', 'rb')
    results = pickle.load(f)

    # print "Enter 0 (or) 1 for without/with background"
    # withBackground = int(input())
    withBackground = 0

    for i in range(len(results)):
        print(i)
        name = 'sequence' + str(i)
        plot_trajectories(results[i][0], results[i][1], results[i][2], results[i][3], name, plot_directory, withBackground)


if __name__ == '__main__':
    main()
