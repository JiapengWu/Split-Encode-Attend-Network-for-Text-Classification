import data_loader as dl
from utility import *
import parameters as p
import torch.nn as nn
import pickle
import bcolz


def gen_2d_mini_batch(tokens, labels, mini_batch_size):
    for token, label in iterate_mini_batches(tokens, labels, mini_batch_size):
        token = pad_2d_batch(token)
        yield token, Variable(torch.from_numpy(label), requires_grad=False)


def test_idx_to_word():
    dataset.text2idx(word2idx)
    g = gen_2d_mini_batch(X_train, y_train, 2)
    for i in range(1):
        try:
            tokens, labels = next(g)
            tokens = tokens.transpose(0, 1).data.numpy().tolist()
            print(list(map(lambda sent: list(map(lambda word: idx2word[word], sent)), tokens)))
        except StopIteration:
            print("Epoch finished")


def test_embed():
    g = gen_2d_mini_batch(X_train, y_train, 2)
    for i in range(1):
        try:
            tokens, labels = next(g)
            embedded = embedding(tokens)

        except StopIteration:
            print("Epoch finished")


def test_embeddings(emb_size):
    embsize_index = {200: '6B.', 300: '840B.'}
    vectors = bcolz.open("{}{}{}.dat".format(p.glove_dir, embsize_index[emb_size], str(emb_size)))[:]
    words = pickle.load(open(("{}{}{}_words.pkl".format(p.glove_dir, embsize_index[emb_size], str(emb_size))), 'rb'))
    embedding_word2idx = pickle.load(open(("{}{}{}_idx.pkl".format(p.glove_dir, embsize_index[emb_size], str(emb_size))), 'rb'))
    glove = {w: vectors[embedding_word2idx[w]] for w in words}
    weights_matrix = np.zeros((len(word2idx.keys()), emb_size))

    words_found = 0
    for word, i in word2idx.items():
        try:
            weights_matrix[i] = glove[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_size,))
    print("{} words have been found".format(words_found))
    return glove, weights_matrix


if __name__ == '__main__':
    dataset, word2idx, idx2word = dl.load_data_set()
    print(dataset.X_train[:2].values)
    dataset.text2idx(word2idx)
    X_train, y_train, X_val, y_val, X_test, y_test = \
        dataset.X_train, dataset.y_train, dataset.X_val, dataset.y_val, dataset.X_test, dataset.y_test

    glove, weights_matrix = test_embeddings(300)
    for i in range(len(word2idx)):
        word = idx2word[i]
        vect_1 = weights_matrix[i]
        try:
            vect_2 = glove[idx2word[i]]
            assert np.array_equal(vect_1, vect_2)
        except:
            print("Unk: {}".format(word))

    weights_matrix = torch.from_numpy(weights_matrix).float()
    embedding = nn.Embedding(weights_matrix.shape[0],  weights_matrix.shape[1])
    embedding.weight = nn.Parameter(weights_matrix, requires_grad=True)
    test_idx_to_word()