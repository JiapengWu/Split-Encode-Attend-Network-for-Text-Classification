import time
import math
import utility
import os
from utility import get_args
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import numpy as np
import pandas as pd
from dataset import TextDataset
from hierarchical_models import *
from word_seq_models import *
from data_loader import construct_dictionary
import parameters as p
import sys


def process_batch(tokens, labels):
    tokens = utility.pad_3d_batch(tokens)
    return tokens.cuda(), Variable(torch.from_numpy(labels), requires_grad=False).cuda()


def check_loss_and_accuracy(grouped):
    loss_list = []
    preds = []
    label_list = []
    for name, group in grouped:
        tokens = TextDataset._text2idx(group.tokens, dictionary.word2idx)
        labels = np.array(group.label.values)
        tokens, labels = process_batch(tokens, labels)
        if config.para_pooling == 'attn':
            y_pred, _ = model.forward(tokens)
        else:
            y_pred = model.forward(tokens)

        labels = labels.view(labels.shape[0], -1)
        loss = criterion(y_pred.cuda(), labels[0])
        loss_list.append(loss.item())

        _, y_pred = torch.max(y_pred, 1)
        preds.append(y_pred.item())
        label_list.append(labels[0].item())
    preds = np.array(preds)
    label_list = np.array(label_list)
    precision, recall, f1, _ = precision_recall_fscore_support(label_list, preds, average='macro')
    return np.mean(np.array(loss_list)), accuracy_score(label_list, preds), precision, recall, f1, confusion_matrix(label_list, preds)


def train_data(mini_batch, targets):
    model.train()

    optimizer.zero_grad()
    if config.para_pooling == 'attn':
        y_pred, _ = model.forward(mini_batch)
    else:
        y_pred = model.forward(mini_batch)

    targets = targets.view(targets.shape[0], -1)
    loss = criterion(y_pred.cuda(), targets[0])
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), config.clip)
    optimizer.step()
    return loss.item()


def train_early_stopping(epoch_number):
    global best_val_loss, best_acc

    loss_epoch = []
    i = 1
    batch_start = time.time()
    for name, group in train_grouped:
        # print(group.tokens.values)
        tokens = TextDataset._text2idx(group.tokens, dictionary.word2idx)
        labels = np.array(group.label.values)
        try:
            tokens, labels = process_batch(tokens, labels)
        except:
            print(tokens)
            sys.exit(0)

        loss = train_data(tokens, labels)
        loss_epoch.append(loss)
        # print loss every n passes
        if i % (p.print_loss_every * 5) == 0:
            print('| epoch   %d | %d/%d batches | ms/batch (%s) | loss %f' % (
                epoch_number, i % (num_batches + 1), num_batches, time_since(batch_start), np.mean(loss_epoch)))
            batch_start = time.time()
        i += 1

    # word_encoder.eval()
    # sent_encoder.eval()
    model.eval()

    print('-' * 89)
    val_loss, val_acc, precision, recall, f1, conf_matrix = check_loss_and_accuracy(val_grouped)
    print('| val set result | valid loss (pure) {:5.4f} | Acc {:8.4f} | Precision {:8.4f} | Recall {:8.4f} '
          '| F1-score {:8.4f}'.format(val_loss, val_acc, precision, recall, f1))
    print('The confusion matrix is: ')
    print(str(conf_matrix))
    print('-' * 89)

    test_loss, test_acc, precision, recall, f1, conf_matrix = check_loss_and_accuracy(test_grouped)
    print('| test set result | valid loss (pure) {:5.4f} | Acc {:8.4f} | Precision {:8.4f} | Recall {:8.4f} '
          '| F1-score {:8.4f}'.format(test_loss, test_acc, precision, recall, f1))
    print('The confusion matrix is: ')
    print(str(conf_matrix))
    print('-' * 89)

    directory = "./experiments/%s/models/" % config.exp_num

    if not os.path.exists(directory):
        os.makedirs(directory)

    if not best_val_loss or val_loss < best_val_loss:
        best_val_loss = val_loss
    else:  # if loss doesn't go down, divide the learning rate by 5.
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.2
    if not best_acc or val_acc > best_acc:
        with open(directory + 'para_{}.best_acc.pt'.format(config.para_pooling), 'wb') as f:
            torch.save(model, f)
        best_acc = val_acc
    with open(directory + 'para_{}.epoch-{:02d}.pt'.format(config.para_pooling, epoch_number), 'wb') as f:
        torch.save(model, f)

    with open("./experiments/{}/optimizer.pt".format(config.exp_num), 'wb') as f:
        torch.save(optimizer.state_dict(), f)


def init_recurrent_model(config, word2idx):
    if config.word_pooling == 'attn':
        word_encoder = AttentionWordRNN(word2idx, config)
    else:
        word_encoder = WordEncoder(word2idx, config)

    if config.sent_pooling == 'attn':
        sent_encoder = AttentionSentRNN(config)
    else:
        sent_encoder = SentEncoder(config)

    if config.load_model:
        if os.path.isfile('./log_exp_{}'.format(config.model_exp_num)):
            log_file = './log_exp_{}'.format(config.model_exp_num)
        else:
            log_file = 'logs/log_exp_{}'.format(config.model_exp_num)
        val_set = []
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'val set' in line:
                    val_set.append(line)

        f1s = [float(line[-6:]) for line in val_set]
        index = np.argmax(f1s)

        print("loading model {} at number {} epoch".format(config.model_exp_num, index + 1))
        pre_word_encoder = torch.load("./experiments/{}/models/word_attn.epoch-{:02d}.pt".format(config.model_exp_num, index))
        pre_sent_encoder = torch.load("./experiments/{}/models/sent_attn.epoch-{:02d}.pt".format(config.model_exp_num, index))

        word_encoder.load_state_dict(pre_word_encoder)
        sent_encoder.load_state_dict(pre_sent_encoder)

    if not config.tune_model:
        for para in word_encoder.parameters():
            para.requires_grad = False
        for para in sent_encoder.parameters():
            para.requires_grad = False

    return word_encoder.cuda(), sent_encoder.cuda()



def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def load_data():
    if p.three_parts_split:
        data_train = pd.read_json(p.sent_split_dir + "train_3_parts_split.json")
        data_val = pd.read_json(p.sent_split_dir + "val_3_parts_split.json")
        data_test = pd.read_json(p.sent_split_dir + "test_3_parts_split.json")
    else:
        if config.word_seq:
            data_train = pd.read_json(p.word_seq_dir + "train_paragraph_word.json")
            data_val = pd.read_json(p.word_seq_dir + "val_paragraph_word.json")
            data_test = pd.read_json(p.word_seq_dir + "test_paragraph_word.json")
        else:
            data_train = pd.read_json(p.sent_split_dir + "train_paragraph_sentence.json")
            data_val = pd.read_json(p.sent_split_dir + "val_paragraph_sentence.json")
            data_test = pd.read_json(p.sent_split_dir + "test_paragraph_sentence.json")

    if config.dataset == "yelp":
        data_train['label'] = data_train['label'].apply(lambda x: 0 if x == 1 else 1).astype('int64')
        data_val['label'] = data_val['label'].apply(lambda x: 0 if x == 1 else 1).astype('int64')
        data_test['label'] = data_test['label'].apply(lambda x: 0 if x == 1 else 1).astype('int64')

    if p.small_data:
        data_train = data_train.head(50)
        data_val = data_val.head(50)
        data_test = data_test.head(50)
    return data_train, data_val, data_test


if __name__ == '__main__':
    config = get_args()
    data_train, data_val, data_test = load_data()
    config.class_number = data_train['label'].nunique()
    config.levels = 3
    print(config)
    train_grouped = data_train.groupby("id")
    val_grouped = data_val.groupby("id")
    test_grouped = data_test.groupby("id")
    num_batches = len(train_grouped)

    if config.build_vocab:
        X_train = data_train.tokens
        X_val = data_val.tokens
        X_test = data_test.tokens
        dictionary = construct_dictionary(X_train, X_val, X_test)
    else:
        dictionary = pd.read_pickle(p.dict_path)
    if config.para_pooling == 'cnn':
        raise Exception("Please run 2_level_model_tune.py")
    else:
        word_encoder, sent_encoder = init_recurrent_model(config, dictionary.word2idx)

    cur_epoch = 1
    if config.resume:
        from functools import reduce
        import re
        from os import listdir
        from os.path import isfile, join

        mypath = "./experiments/" + str(config.exp_num) + "/models/"
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        try:
            cur_epoch = 1 + int(max(list(reduce(lambda x, y: x + y, [re.findall(r'\d+', x) for x in onlyfiles]))))
            print("Continuing training...")
            model = torch.load("./experiments/" + str(config.exp_num) + "/models/"
                                        + 'para_{}.epoch-{:02d}.pt'.format(config.para_pooling, cur_epoch - 1))
        except ValueError:
            cur_epoch = 1

    else:
        # if config.para_pooling == 'attn':
        #     model = ParagraphAttention(config, word_encoder, sent_encoder).cuda()
        # elif config.para_pooling == 'ensem':
        #     model = ParagraphEnsemble(config, word_encoder, sent_encoder).cuda()
        # elif config.para_pooling == 'ensem-attn':
        #     model = ParagraphEnsembleAttention(config, word_encoder, sent_encoder).cuda()
        # else:
        #     model = ParagraphPooling(config, word_encoder, sent_encoder).cuda()
        model = ParagraphAttention(config, word_encoder, sent_encoder).cuda()

    print(model)

    if config.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)
    elif config.optimizer == "SGD":
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr, momentum=config.momentum)
    criterion = nn.NLLLoss()

    if config.resume:
        pre_optim = torch.load("./experiments/{}/optimizer.pt".format(config.exp_num))
        optimizer.load_state_dict(pre_optim)

    best_val_loss = None
    best_acc = None

    for i in range(cur_epoch, p.num_epoch + 1):
        train_early_stopping(i)