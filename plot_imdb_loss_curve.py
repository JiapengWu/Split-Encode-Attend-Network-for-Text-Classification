
from textwrap import wrap
import os
from utility import *
import sys
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt


def plot_loss_acc():
    plt.figure(1)

    plt.subplot(311)
    plt.plot(np.arange(len(train_loss)), train_loss, "b", lw=2, marker='.')
    plt.legend(['training loss'], loc='upper right')
    plt.title('loss and acc')
    plt.grid(True)
    plt.show()

    plt.subplot(312)
    plt.plot(np.arange(len(val_loss)), val_loss, "coral", lw=2, marker='.')
    plt.legend(['val loss'], loc='upper right')
    plt.grid(True)
    plt.show()

    plt.subplot(313)
    plt.plot(np.arange(len(val_acc)), val_acc, "coral", lw=2, marker='.')
    plt.legend(['val acc'], loc='upper right')
    plt.grid(True)
    plt.show()

    fname = "loss and acc exp {}".format(exp_num)
    plt.savefig('plots/{}.png'.format(fname))
    plt.clf()


if __name__ == '__main__':

    exp_num = sys.argv[1]

    if os.path.isfile('./log_exp_{}'.format(exp_num)):
        log_file = './log_exp_{}'.format(exp_num)
    else:
        log_file = 'logs/log_exp_{}'.format(exp_num)
    train_loss = []
    val_loss = []
    val_acc = []

    with open(log_file, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            line = lines[i]
            try:
                next_line = lines[i + 1]
            except IndexError:
                break

            # if "Namespace" in line:
            #     print(line)
            if 'val set' in line:
                val_loss.append(float(line.split()[8]))
                val_acc.append(float(line.split()[11]))
            if '----------' in next_line and 'epoch' in line:
                train_loss.append(float(line.split()[-1]))
    # print(len(val_acc), len(val_loss), len(train_loss))
    plot_loss_acc()


