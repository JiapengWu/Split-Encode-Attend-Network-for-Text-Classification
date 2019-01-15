from utility import Dictionary, get_args
from dataset import TextDataset
import pickle
import parameters as p
import numpy as np
import bcolz
import pandas as pd

args = get_args()

def load_data_set():
    dataset = TextDataset()
    print("finished reading datasets")
    if not args.build_vocab:
        with open(p.dict_path, 'rb') as f:
            dictionary = pickle.load(f)
    else:
        dictionary = construct_dictionary(dataset.X_train, dataset.X_val, dataset.X_test)
    return dataset, dictionary.word2idx, dictionary.idx2word


def construct_dictionary(data_train, data_val, data_test):
    if args.use_val:
        dataset = pd.concat([data_train, data_val], 0)
        if args.use_test:
            dataset = pd.concat([dataset, data_test], 0)
    else:
        dataset = data_train
    print("constructing doctionary...")
    dictionary = Dictionary()
    dictionary.word2idx, dictionary.idx2word = TextDataset.assign_word_ids(args.emsize, dataset)
    print("----processed {%d} word_2_id----" % (len(dictionary.word2idx)))
    with open(p.dict_path, 'wb') as f:
        pickle.dump(dictionary, f)
    return dictionary


def assign_embedding(word2idx, emb_size):
    if not args.build_embedding:
        weights_matrix = np.load(p.glove_matrix_path)
    else:
        embsize_index = {200: '6B.', 300: '840B.'}
        vectors = bcolz.open(p.glove_dir + embsize_index[emb_size] + str(emb_size) + '.dat')[:]
        words = pickle.load(open(p.glove_dir + embsize_index[emb_size] + str(emb_size) + '_words.pkl', 'rb'))
        embedding_word2idx = pickle.load(open(p.glove_dir + embsize_index[emb_size] + str(emb_size) + '_idx.pkl', 'rb'))

        glove = {w: vectors[embedding_word2idx[w]] for w in words}

        weights_matrix = np.zeros((len(word2idx.keys()), emb_size))
        words_found = 0
        for word, i in word2idx.items():
            try:
                weights_matrix[i] = glove[word]
                words_found += 1
            except KeyError:
                weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_size,))
        print(str(words_found) + " words have been found")

        np.save(p.glove_matrix_path, weights_matrix)
    return weights_matrix


if __name__ == "__main__":
    dataset, word2idx, idx2word = load_data_set()
    assign_embedding(word2idx, get_args().emsize)
