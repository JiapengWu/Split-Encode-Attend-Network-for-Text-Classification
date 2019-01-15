import numpy as np
import torch
from torch.autograd import Variable
import json
import argparse


class Dictionary(object):
    def __init__(self, path=''):
        self.word2idx = dict()
        self.idx2word = dict()
        if path != '':  # load an external dictionary
            words = json.loads(open(path, 'r').readline())
            for item in words:
                self.add_word(item)

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = len(self.word2idx)
            self.idx2word[len(self.idx2word)] = word
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser()

    # -------------------------------- Meta parameters --------------------------------------
    parser.add_argument('--dataset', type=str, default='sent',
                        help='which dataset to use. ling, auth, sent or topic')
    parser.add_argument('--use-paragraph', type=str2bool, default=True,
                        help='whether using paragraph dataset')
    parser.add_argument('--fixed-length', type=str2bool, default=False,
                        help='whether using fixed-length paragraph dataset')
    parser.add_argument('--word-seq', type=str2bool, default=False,
                        help='whether using word sequence data')
    parser.add_argument('--build-vocab', type=str2bool, default=False,
                        help='build vocab')
    parser.add_argument('--build-embedding', type=str2bool, default=False,
                        help='build embeddings')
    parser.add_argument('--use-glove', type=str2bool, default=True,
                        help='use glove or not')
    parser.add_argument('--original-split', type=str2bool, default=True,
                        help='original split')
    parser.add_argument('--debug', type=str2bool, default=False,
                        help='entering debugging mode')
    parser.add_argument('--optimizer', type=str, default='SGD',
                        help='type of optimizer')
    parser.add_argument('--first-k-sents', type=int, default=999,
                        help='use first k sentences')
    parser.add_argument('--first-k-paras', type=int, default=999,
                        help='use first k paragraphs')
    # -------------------------------- model training and testing --------------------------------------
    parser.add_argument('--emsize', type=int, default=300,
                        help='size of word embeddings')
    parser.add_argument('--word-gru-hidden', type=int, default=50,
                        help='number of hidden units per layer')
    parser.add_argument('--sent-gru-hidden', type=int, default=50,
                        help='number of hidden units per layer')
    parser.add_argument('--para-gru-hidden', type=int, default=50,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers in BiLSTM')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size for paragraph model training')
    parser.add_argument('--encoder-type', type=str, default='LSTM',
                        help='LSTM, GRU or CNN')
    parser.add_argument('--levels', type=int, default=3,
                        help='Number of encoder levels')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--num-epoch', type=int, default=20,
                        help='number of epochs.')
    parser.add_argument('--clip', type=float, default=0.5,
                        help='clip to prevent the too large grad in LSTM')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='initial momentum')
    parser.add_argument('--class-number', type=int, default=2,
                        help='number of classes')
    parser.add_argument('--resume', type=str2bool, default=False,
                        help='loading the existing model and continue training')
    parser.add_argument('--use-val', type=str2bool, default=False,
                        help='use the concatenation of training and val set as training set.')
    parser.add_argument('--use-test', type=str2bool, default=False,
                        help='use test set when constructing the dictionary.')
    parser.add_argument('--word-model', type=str, default='cnn')

    # -------------------------------- Encoder and saving parameter --------------------------------------
    parser.add_argument('--exp-num', type=int, default=99,
                        help='experiment number for easy track experiments')
    # -------------------------------- paragraph model parameters --------------------------------------
    parser.add_argument('--para-pooling', type=str, default='attn',
                        help='paragraph level pooling. attn for attention, ensem for ensemble, ensem-attn for combination'
                             'cnn for convolutional neural net, trans for transformer, multi-hop for multihop net'
                             'mean or mean pooling, max for max pooling')
    parser.add_argument('--word-pooling', type=str, default='attn',
                        help='word level pooling, attn for attention, mean or mean pooling, max for max pooling')
    parser.add_argument('--sent-pooling', type=str, default='attn',
                        help='sentence level pooling, attn for attention, mean or mean pooling, max for max pooling')
    parser.add_argument('--load-model', type=str2bool, default=False,
                        help='load pre-trained paragraph classification model')
    parser.add_argument('--model-exp-num', type=int, default=1,
                        help='name of the folder where we load model from')
    parser.add_argument('--tune-model', type=str2bool, default=True,
                        help='train hierarchical attention model')

    # -------------------------------- CNN --------------------------------------
    parser.add_argument('--kernel-num', type=int, default=250,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel-sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--use-pyramid', type=str2bool, default=False,
                        help='whether using pyramid CNN')
    return parser.parse_args()


def iterate_mini_batches(inputs, targets, batchsize):
    assert inputs.shape[0] == targets.shape[0]
    for start_idx in range(0, inputs.shape[0], batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def pad_3d_batch(mini_batch):
    mini_batch_size = len(mini_batch)
    max_sent_len = int(np.max([len(x) for x in mini_batch]))
    max_token_len = int(np.max([len(val) for sublist in mini_batch for val in sublist]))
    main_matrix = np.zeros((mini_batch_size, max_sent_len, max_token_len), dtype=np.int)
    for i in range(main_matrix.shape[0]):
        for j in range(main_matrix.shape[1]):
            for k in range(main_matrix.shape[2]):
                try:
                    main_matrix[i, j, k] = mini_batch[i][j][k]
                except IndexError:
                    main_matrix[i, j, k] = 0
    return Variable(torch.from_numpy(main_matrix).transpose(0, 1))


def pad_2d_batch(mini_batch):
    mini_batch_size = len(mini_batch)
    max_token_len = int(np.max([len(x) for x in mini_batch]))
    main_matrix = np.zeros((mini_batch_size, max_token_len), dtype=np.int)
    for i in range(main_matrix.shape[0]):
        for j in range(main_matrix.shape[1]):
            try:
                main_matrix[i, j] = mini_batch[i][j]
            except IndexError:
                main_matrix[i, j] = 0
    return Variable(torch.from_numpy(main_matrix).transpose(0, 1))


def gen_3d_mini_batch(tokens, labels, mini_batch_size):
    for token, label in iterate_mini_batches(tokens, labels, mini_batch_size):
        token = pad_3d_batch(token)
        yield token.cuda(), Variable(torch.from_numpy(label), requires_grad=False).cuda()


def gen_2d_mini_batch(tokens, labels, mini_batch_size):
    for token, label in iterate_mini_batches(tokens, labels, mini_batch_size):
        token = pad_2d_batch(token)
        yield token.cuda(), Variable(torch.from_numpy(label), requires_grad=False).cuda()

def model_forward(pooling, word_model, sent_model, mini_batch):
    max_sents, batch_size, max_tokens = mini_batch.size()
    s = None
    state_word = word_model.init_hidden(batch_size)
    state_sent = sent_model.init_hidden(batch_size)
    if pooling == "attn":
        for i in range(max_sents):
            _s, state_word, _ = word_model(mini_batch[i, :, :].transpose(0, 1), state_word)
            if (s is None):
                s = _s
            else:
                s = torch.cat((s, _s), 0)
        y_pred, state_sent, _ = sent_model(batch_size, s, state_sent)
    else:
        for i in range(max_sents):
            _s, state_word = word_model(mini_batch[i, :, :].transpose(0, 1), state_word)
            if (s is None):
                s = _s
            else:
                s = torch.cat((s, _s), 0)
        y_pred = sent_model(batch_size, s, state_sent)
    return y_pred
