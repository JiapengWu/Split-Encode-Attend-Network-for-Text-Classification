import torch
import pandas as pd
import utility
import numpy as np
import parameters as p
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from dataset import TextDataset
from hierarchical_models import AttentionSentRNN, AttentionWordRNN, ParagraphEnsemble, ParagraphAttention
from utility import get_args
from torch.autograd import Variable


def test_accuracy_full_batch(X, y, word_attn, sent_attn, average):
    preds = []
    labels = []
    g = utility.gen_3d_mini_batch(X, y, 64)
    for token, label in g:
        y_pred = utility.get_predictions(token.cuda(), word_attn, sent_attn)
        _, y_pred = torch.max(y_pred, 1)
        preds.append(np.ndarray.flatten(y_pred.data.cpu().numpy()))
        labels.append(np.ndarray.flatten(label.data.cpu().numpy()))
    preds = [item for sublist in preds for item in sublist]
    labels = [item for sublist in labels for item in sublist]
    preds = np.array(preds)
    labels = np.array(labels)
    num_correct = sum(preds == labels)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=average)
    return float(num_correct) / len(preds), precision, recall, f1, confusion_matrix(labels, preds)


def load_paragraph_model(exp_num, epoch, word2idx):
    word_attn = AttentionWordRNN(num_tokens=len(word2idx), embed_size=config.emsize,
                                 word_gru_hidden=config.word_gru_hidden, dropout=config.dropout,
                                 word2idx=word2idx, bidirectional=True)

    sent_attn = AttentionSentRNN(sent_gru_hidden=config.sent_gru_hidden,
                                 word_gru_hidden=config.word_gru_hidden,
                                 dropout=config.dropout, n_classes=config.class_number, bidirectional=True)
    if exp_num == 4:
        pre_word_attn_dict = torch.load("./experiments/4/models/word_attn" + '.epoch-{:02d}.pt'.format(epoch)).cuda().state_dict()
        pre_sent_attn_dict = torch.load("./experiments/4/models/sent_attn" + '.epoch-{:02d}.pt'.format(epoch)).cuda().state_dict()
    word_attn.load_state_dict(pre_word_attn_dict)
    sent_attn.load_state_dict(pre_sent_attn_dict)
    return word_attn.cuda(), sent_attn.cuda()


def load_paragraph_data():
    data_val = pd.read_json(p.sent_split_dir + "val_clustered_sent_split.json")
    data_test = pd.read_json(p.sent_split_dir + "test_clustered_sent_split.json")

    X_val = data_val.tokens
    X_test = data_test.tokens

    y_val = np.array(data_val.label.values)
    y_test = np.array(data_test.label.values)

    dictionary = pd.read_pickle(p.dict_path)
    X_val = TextDataset._text2idx(X_val, dictionary.word2idx)
    X_test = TextDataset._text2idx(X_test, dictionary.word2idx)

    return X_val, y_val, X_test, y_test, dictionary


def evaluate_paragrph():
    exp_num = 4
    for epoch in range(12, 19):
        for average in avgs:
            print("Average: %s | At epoch %d: " % (average, epoch))

            # data_train = pd.read_json(p.sent_split_dir + "train_clustered_sent_split.json")

            X_val, y_val, X_test, y_test, dictionary = load_paragraph_data()
            word_attn, sent_attn = load_paragraph_model(exp_num, epoch, dictionary.word2idx)
            word_attn.eval()
            sent_attn.eval()

            """evaluate the model """
            print('-' * 89)
            val_acc, precision, recall, f1, conf_matrix = test_accuracy_full_batch(X_val, y_val, word_attn, sent_attn, average)
            print('| val result | Acc  %f | Precision: %f | Recall: %f | F1-score: %f ' % (val_acc, precision, recall, f1))
            print('The confusion matrix is: ')
            print(str(conf_matrix))
            print('-' * 89)

            print('-' * 89)
            test_acc, precision, recall, f1, conf_matrix = test_accuracy_full_batch(X_test, y_test, word_attn, sent_attn, average)
            print('| test result | Acc  %f | Precision: %f | Recall: %f | F1-score: %f ' % (test_acc, precision, recall, f1))
            print('The confusion matrix is: ')
            print(str(conf_matrix))
            print('-' * 89)


def process_batch(tokens, labels):
    tokens = utility.pad_3d_batch(tokens)
    return tokens.cuda(), Variable(torch.from_numpy(labels), requires_grad=False).cuda()


def check_loss_and_accuracy(grouped, model, dictionary):
    preds = []
    label_list = []
    for name, group in grouped:
        tokens = TextDataset._text2idx(group.tokens, dictionary.word2idx)
        labels = np.array(group.label.values)
        tokens, labels = process_batch(tokens, labels)
        if config.pooling == 'attn':
            y_pred, _, _ = model.forward(tokens)
        elif config.pooling == 'ensem':
            y_pred = model.forward(tokens)

        labels = labels.view(labels.shape[0], -1)

        _, y_pred = torch.max(y_pred, 1)
        preds.append(y_pred.item())
        label_list.append(labels[0].item())
    preds = np.array(preds)
    label_list = np.array(label_list)
    precision, recall, f1, _ = precision_recall_fscore_support(label_list, preds)
    return accuracy_score(label_list, preds), precision, recall, f1, confusion_matrix(label_list, preds)

def evaluate_fist_paragrph(exp_num):
    data_val = pd.read_json(p.sent_split_dir + "val_clustered_sent_split.json")
    data_test = pd.read_json(p.sent_split_dir + "test_clustered_sent_split.json")
    val_grouped = data_val.groupby("id")
    test_grouped = data_test.groupby("id")
    dictionary = pd.read_pickle(p.dict_path)
    model = torch.load("./experiments/" + str(config.exp_num) + "/models/"
                       + 'para_{}.epoch-{:02d}.pt'.format(config.pooling, exp_num))


if __name__ == '__main__':
    config = get_args()
    avgs = ['weighted', 'micro']
    evaluate_paragrph()
