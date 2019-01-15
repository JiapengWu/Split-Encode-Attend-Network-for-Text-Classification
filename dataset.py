from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import parameters as p
import pickle
from collections import Counter
import ast
from utility import get_args


class TextDataset(Dataset):
    # Initialize your data, download, etc.

    def __init__(self):
        args = get_args()

        if args.use_paragraph:
            if p.three_parts_split:
                data_train = pd.read_json(p.sent_split_dir + "train_3_parts_split.json")
                data_val = pd.read_json(p.sent_split_dir + "val_3_parts_split.json")
                data_test = pd.read_json(p.sent_split_dir + "test_3_parts_split.json")
            else:
                if args.word_seq:
                    data_train = pd.read_json(p.word_seq_dir + "train_paragraph_word.json")
                    data_val = pd.read_json(p.word_seq_dir + "val_paragraph_word.json")
                    data_test = pd.read_json(p.word_seq_dir + "test_paragraph_word.json")
                else:
                    data_train = pd.read_json(p.sent_split_dir + "train_paragraph_sentence.json")
                    data_val = pd.read_json(p.sent_split_dir + "val_paragraph_sentence.json")
                    data_test = pd.read_json(p.sent_split_dir + "test_paragraph_sentence.json")
        else:
            if args.word_seq:
                data_train = pd.read_json(p.word_seq_dir + "train_document_word.json")
                data_val = pd.read_json(p.word_seq_dir + "val_document_word.json")
                data_test = pd.read_json(p.word_seq_dir + "test_document_word.json")
            else:
                data_train = pd.read_json(p.sent_split_dir + "train_document_sentence.json")
                data_val = pd.read_json(p.sent_split_dir + "val_document_sentence.json")
                data_test = pd.read_json(p.sent_split_dir + "test_document_sentence.json")

        if p.small_data:
            data_train = data_train.head(50)
            data_val = data_val.head(50)
            data_test = data_test.head(50)

        self.X_train = data_train.tokens
        self.X_val = data_val.tokens
        self.X_test = data_test.tokens

        self.y_train = np.array(data_train.label.values)
        self.y_val = np.array(data_val.label.values)
        self.y_test = np.array(data_test.label.values)

        if args.dataset == "yelp" :
            for labels in [self.y_train, self.y_val, self.y_test]:
                labels[labels == 1] = 0
                labels[labels == 2] = 1
        if args.dataset == "ag_news" :
            for labels in [self.y_train, self.y_val, self.y_test]:
                for label in list(range(4)):
                    labels[labels == label + 1] = label


    @staticmethod
    def assign_word_ids(emb_size, df_texts, special_tokens=["<pad>", "<unk>"]):
        """
        Given df_texts (list of sent tokens), create word2id and id2word
        based on the most common words
        :param  df_text: list of sent tokens
        :param special_tokens: set of special tokens to add into dictionary
        :param vocab_size: max_number of vocabs
        :return: word2id, id2word
        """
        args = get_args()
        id = 0
        word2id = {}
        # add special tokens in w2i
        for tok in special_tokens:
            word2id[tok] = id
            id += 1
            print(tok, word2id[tok])
        word_set = [word for doc in df_texts for sent in doc for word in sent.split()]
        # elif p.model_type == 'word':
        #     word_set = [word for doc in df_texts for word in doc]
        c = Counter(word_set)

        ## if max_vocab is not -1, then shrink the word size
        train_words = list(c.keys())
        if args.use_glove:
            embsize_index = {200: '6B.', 300: '840B.'}
            glove_words = pickle.load(open(p.glove_dir + embsize_index[emb_size] + str(emb_size) + '_idx.pkl', 'rb'))
            # unks are the words that have <= 5 frequency and NOT found in gloves
            unks = [word for word in train_words if c[word] <= 5]
            unks = list(set(unks)-set(glove_words))
        else:
            unks = [word for word in train_words if c[word] <= 5]
        # print(unks)
        print("Number of unks: " + str(len(unks)))

        vocab = list(set(train_words) - (set(unks)))
        # add regular words in
        for word in vocab:
            word2id[word] = id
            id += 1
        id2word = {v: k for k, v in word2id.items()}
        # print('finishing processing %d vocabs' % len(word2id))
        return word2id, id2word

    @staticmethod
    def _text2idx(df_texts, word2idx):
        args = get_args()
        if args.use_paragraph:
            if args.word_seq:
                return np.array(list(map(lambda paragraph: list(map(lambda word: word2idx[word] if word in word2idx.keys()
                else word2idx['<unk>'], paragraph)), df_texts)))

            else: return np.array(list(map(lambda paragraph: list(map(lambda sent: list(map(
                lambda word: word2idx[word] if word in word2idx.keys()
                else word2idx['<unk>'], sent.split())), paragraph[:args.first_k_paras])), df_texts)))
        else:
            if args.word_seq:
                return np.array(
                    list(map(lambda paragraph: list(map(lambda word: word2idx[word] if word in word2idx.keys()
                    else word2idx['<unk>'], paragraph)), df_texts)))
            else:
                return np.array(list(map(lambda doc: list(map(lambda sent: list(map(
                    lambda word: word2idx[word] if word in word2idx.keys()
                    else word2idx['<unk>'], sent.split())), doc[:args.first_k_sents])), df_texts)))

    def text2idx(self, word2idx):
        self.X_train = self._text2idx(self.X_train, word2idx)
        self.X_val = self._text2idx(self.X_val, word2idx)
        self.X_test = self._text2idx(self.X_test, word2idx)
        # print(min(min(min(self.X_train.tolist()))))


if __name__ == '__main__':
    TextDataset()
    # TextDataset.load_glove_model()
    # TextDataset