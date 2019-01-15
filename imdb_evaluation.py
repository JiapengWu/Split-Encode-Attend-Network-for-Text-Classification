import numpy as np
import torch
import parameters as p
import pandas as pd
from dataset import TextDataset
from utility import *
import pickle
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from textwrap import wrap


def load_val_test_data():
    data_val = pd.read_json(p.sent_split_dir + "val_paragraph_sentence.json")
    data_test = pd.read_json(p.sent_split_dir + "test_paragraph_sentence.json")
    return data_val, data_test


def get_prediction_slength(grouped):
    preds = []
    label_list = []
    slength_list = []
    # for every batch
    for name, group in grouped:
        s = np.sum(list(map(lambda x: len(x), group.tokens)))
        slength_list.append(s)
        tokens = TextDataset._text2idx(group.tokens, dictionary.word2idx)
        labels = np.array(group.label.values)
        tokens, labels = process_batch(tokens, labels)
        if config.pooling == 'attn':
            y_pred, _, _ = model.forward(tokens)
        else:
            y_pred = model.forward(tokens)
        _, y_pred = torch.max(y_pred, 1)
        preds.append(y_pred.item())
        label_list.append(labels[0].item())
    return preds, label_list, slength_list


def calculate_accuracy_for_slength(grouped, val_or_test):
    preds, labels, slengths = get_prediction_slength(grouped)
    results = np.equal(preds, labels)

    slength_set = set(slengths)
    slen_result = list(map(lambda x: list(x), zip(slengths, results)))
    result_per_length = {x: [y[1] for y in slen_result if y[0] == x] for x in slength_set}
    slen = list(slength_set)
    acc = list(map(lambda v: np.sum(v)/len(v), result_per_length.values()))
    with open("./experiments/{}/{}_accs.p".format(config.exp_num, val_or_test), "wb") as fp:
        pickle.dump(acc, fp)
    with open("./experiments/{}/{}_slen.p".format(config.exp_num, val_or_test), "wb") as fp:
        pickle.dump(slen, fp)
    return slen, acc


def process_batch(tokens, labels):
    tokens = pad_3d_batch(tokens)
    return tokens.cuda(), Variable(torch.from_numpy(labels), requires_grad=False).cuda()


def load_model():
    log_file = 'logs/log_exp_{}'.format(exp_num)
    val_set = []
    with open(log_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'val set' in line:
                val_set.append(line)

    f1s = [float(line[-6:]) for line in val_set]
    index = np.argmax(f1s)

    model = torch.load("./experiments/{}/models/para_attn.epoch-{:02d}.pt".format(exp_num, index + 1))

    return model.cuda().eval()


def plot_accuracy_for_length(x, y, xlabel=None, ylabel="accuracy", val_or_test='val'):

    plt.plot(x, y, lw=2)
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
    model = load_model()
    data_val, data_test = load_val_test_data()
    dictionary = pd.read_pickle(p.dict_path)
    config = model.config
    val_grouped = data_val.groupby("id")
    test_grouped = data_test.groupby("id")

    slen, acc = calculate_accuracy_for_slength(val_grouped, 'val')
    plot_accuracy_for_length(slen, acc, xlabel='sentences', val_or_test='val')
    slen, acc = calculate_accuracy_for_slength(test_grouped, 'test')
    plot_accuracy_for_length(slen, acc, xlabel='sentences', val_or_test='test')