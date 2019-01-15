import torch
import torch.nn as nn
from torch.autograd import Variable
import time
import math
import utility
import parameters as p
import os
from utility import get_args
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import numpy as np
import pandas as pd
from dataset import TextDataset
from hierarchical_models import ParagraphEnsemble, ParagraphAttention, AttentionWordRNN, AttentionSentRNN


def process_batch(tokens, labels):
    tokens = utility.pad_3d_batch(tokens)
    return tokens.cuda(), Variable(torch.from_numpy(labels), requires_grad=False).cuda()


def check_loss_and_accuracy(grouped):
    loss = []
    preds = []
    labels = []
    for name, group in grouped:
        tokens = TextDataset._text2idx(group.tokens, dictionary.word2idx)
        labels = np.array(group.label.values)
        tokens, labels = process_batch(tokens, labels)
        y_pred = model.forward(tokens)
        loss.append(loss.item())

        loss = criterion(y_pred.cuda(), labels[0])
        _, y_pred = torch.max(y_pred, 1)
        preds.append(np.ndarray.flatten(y_pred.data.cpu().numpy()))
        labels.append(np.ndarray.flatten(labels[0]))
    preds = np.array([item for sublist in preds for item in sublist])
    labels = np.array([item for sublist in labels for item in sublist])
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds)
    return np.mean(np.array(loss)), accuracy_score(labels, preds), precision, recall, f1, confusion_matrix(labels, preds)


def train_data(mini_batch, targets):
    model.train()
    if config.tune_model:
        word_attn.train()
        sent_attn.train()
    else:
        word_attn.eval()
        sent_attn.eval()

    if config.tune_model:
        word_optimizer.zero_grad()
        sent_optimizer.zero_grad()
    model_optimizer.zero_grad()

    max_sents, batch_size, max_tokens = mini_batch.size()
    state_word = word_attn.init_hidden(batch_size).cuda()
    state_sent = sent_attn.init_hidden(batch_size).cuda()

    # disable dropout if necessary
    s = None
    for i in range(max_sents):
        _s, state_word, _ = word_attn(mini_batch[i, :, :].transpose(0, 1), state_word)
        if (s is None):
            s = _s
        else:
            s = torch.cat((s, _s), 0)

    y_pred, state_sent, sent_attn_vectors = sent_attn(batch_size, s, state_sent)  # sent_attn_vectors: [plen, nhid*2]

    if config.pooling == 'attn':
        state_para = model.init_hidden().cuda()
        y_pred = model.forward(sent_attn_vectors, state_para)
    elif config.pooling == 'ensem':
        y_pred = model.forward(y_pred, sent_attn_vectors)

    targets = targets.view(targets.shape[0], -1)
    loss = criterion(y_pred.cuda(), targets[0])

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), p.clip)

    if config.tune_model:
        word_optimizer.step()
        sent_optimizer.step()
    model_optimizer.step()

    return loss.item()


def train_early_stopping(epoch_number):
    global best_val_loss, best_acc
    start = time.time()
    loss_epoch = []
    i = 1
    batch_start = time.time()
    for name, group in train_grouped:
        # print(group.tokens.values)
        tokens = TextDataset._text2idx(group.tokens, dictionary.word2idx)
        labels = np.array(group.label.values)
        tokens, labels = process_batch(tokens, labels)

        loss = train_data(tokens, labels)
        loss_epoch.append(loss)
        # print loss every n passes
        if i % (p.print_loss_every * 5) == 0:
            print('| epoch   %d | %d/%d batches | ms/batch (%s) | loss %f' % (
                epoch_number, i % (num_batches + 1), num_batches, time_since(batch_start), np.mean(loss_epoch)))
            batch_start = time.time()
        i += 1

    word_attn.eval()
    sent_attn.eval()
    model.eval()
    print('-' * 89)
    val_loss, val_acc, precision, recall, f1, conf_matrix = check_loss_and_accuracy(val_grouped)
    print('| val set loss  %f | time  %s | Acc  %f' % (val_loss, time_since(start), val_acc) +
          "| Precision: " + str(precision) + " | Recall: " + str(recall) + " | F1-score: " + str(f1))
    print('The confusion matrix is: ')
    print(str(conf_matrix))
    print('-' * 89)

    test_loss, test_acc, precision, recall, f1, conf_matrix = check_loss_and_accuracy(test_grouped)
    print('| test set loss:  %f| Acc  %f ' % (test_loss, test_acc) +
          "| Precision: " + str(precision) + " | Recall: " + str(recall) + " | F1-score: " + str(f1))
    print('The confusion matrix is: ')
    print(str(conf_matrix))
    print('-' * 89)

    directory = "./experiments/%s/models/" % config.exp_num

    if not os.path.exists(directory):
        os.makedirs(directory)

    if not best_val_loss or val_loss < best_val_loss:
        best_val_loss = val_loss
    else:  # if loss doesn't go down, divide the learning rate by 5.
        for param_group in model_optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.2
    if not best_acc or val_loss > best_acc:
        with open(directory + p.para_ensem_path[:-3] + '.best_acc.pt', 'wb') as f:
            torch.save(model, f)
        best_acc = val_loss
    with open(directory + p.para_ensem_path[:-3] +
                      '.epoch-{:02d}.pt'.format(epoch_number), 'wb') as f:
        torch.save(model, f)


def init_model(config, word2idx, epoch):
    if config.load_model:
        word_attn = torch.load('./experiments/{}/models/word_attn.epoch-{:02d}.pt'.format(config.model_exp_num, epoch))
        sent_attn = torch.load('./experiments/{}/models/sent_attn.epoch-{:02d}.pt'.format(config.model_exp_num, epoch))
    else:
        word_attn = AttentionWordRNN(num_tokens=len(word2idx), embed_size=config.emsize,
                                          word_gru_hidden=config.word_gru_hidden, dropout=config.dropout,
                                          word2idx=word2idx, bidirectional=True)

        sent_attn = AttentionSentRNN(sent_gru_hidden=config.sent_gru_hidden,
                                          word_gru_hidden=config.word_gru_hidden,
                                          dropout=config.dropout, n_classes=config.class_number, bidirectional=True)

    return word_attn.cuda(), sent_attn.cuda()


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


if __name__ == '__main__':
    config = get_args()
    print(config)
    data_train = pd.read_json(p.sent_split_dir + "train_clustered_sent_split.json")
    data_val = pd.read_json(p.sent_split_dir + "val_clustered_sent_split.json")
    data_test = pd.read_json(p.sent_split_dir + "test_clustered_sent_split.json")
    train_grouped = data_train.groupby("id")
    val_grouped = data_val.groupby("id")
    test_grouped = data_test.groupby("id")
    num_batches = len(train_grouped)

    dictionary = pd.read_pickle(p.dict_path)
    word_attn, sent_attn = init_model(config, dictionary.word2idx, 17)
    if config.pooling == 'attn':
        model = ParagraphAttention(config).cuda()
    elif config.pooling == 'ensem':
        model = ParagraphEnsemble(config).cuda()
    print(model)

    if config.tune_model:
        word_optimizer = torch.optim.SGD(word_attn.parameters(), lr=config.lr, momentum=config.momentum)
        sent_optimizer = torch.optim.SGD(sent_attn.parameters(), lr=config.lr, momentum=config.momentum)
    model_optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)

    criterion = nn.NLLLoss()

    for i in range(1, p.num_epoch + 1):
        train_early_stopping(i)

