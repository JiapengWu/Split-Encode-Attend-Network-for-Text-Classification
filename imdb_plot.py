import numpy as np
import torch
import pandas as pd
from utility import *
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from textwrap import wrap


def get_vars(val_or_test):
    acc = pd.read_pickle("./experiments/{}/{}_accs.p".format(exp_num, val_or_test))
    slen = pd.read_pickle("./experiments/{}/{}_slen.p".format(exp_num, val_or_test))
    return slen, acc


def plot_accuracy_for_length(x, y, xlabel=None, ylabel="accuracy", val_or_test='val'):

    plt.plot(x, y, lw=2, marker='o')
    plt.axis([np.min(x), np.max(x), 0, 1.1])
    if xlabel:
        plt.xlabel("number of {}".format(xlabel))
    plt.ylabel(ylabel)

    title = "{} set result of 3-level attention network using {}, " \
            "with{} supervised pretraining using {} and dropout {}" \
        .format(val_or_test, "fixed length partition" if config.fixed_length else "GraphSeg partitions",
                '' if config.load_model else "out", config.encoder_type, config.dropout)

    plt.title("\n".join(wrap(title, 60)))
    plt.grid(True)
    fname = "IMdb {} set result, exp {}".format(val_or_test, exp_num)
    plt.savefig('plots/{}.png'.format(fname))
    plt.clf()


if __name__ == '__main__':
    exp_num = get_args().exp_num
    config = torch.load("./experiments/{}/models/para_attn.epoch-01.pt".format(exp_num)).config
    slen, acc = get_vars('val')
    plot_accuracy_for_length(slen, acc, xlabel='sentences', val_or_test='val')
    slen, acc = get_vars('test')
    plot_accuracy_for_length(slen, acc, xlabel='sentences', val_or_test='test')