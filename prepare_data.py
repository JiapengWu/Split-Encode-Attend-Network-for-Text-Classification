from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import torch
import nltk
import json

from nltk.tokenize import stanford
import csv
from nltk.stem import WordNetLemmatizer
from copy import deepcopy
from collections import Counter
import parameters as p
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class_dict = {
        "typical": 0,
        "good": 1,
        "great": 2
    }
documents = {"great" : [],
             "good" : [],
             "typical" : []}
word_count = Counter()

base = '/home/paulwu/research/processed_nyt/'
lemmatizer = WordNetLemmatizer()
sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
work_tokenizer = stanford.StanfordTokenizer(path_to_jar="/home/paulwu/nltk_data/stanford-postagger-2018-02-27/stanford-postagger-3.9.1.jar")


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.index2word = {1: "UNK"}
        self.n_words = 1  # Count UNK

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            if(word_count[word] > 5):
                self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.n_words += 1
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word


eng = Lang("eng")


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" ", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def indexes_from_sentence(sentence):
    indexes = []
    for word in sentence.split(' '):
        try:
            indexes.append(eng.word2index[word])
        except KeyError:
            indexes.append(1)
    return indexes


def preprocess_and_construct_vocab(class_label):
    with open(class_label + '.csv', "r") as f:
        reader = csv.reader(f)
        count = 0
        for row in reader:
            # count += 1
            # if count == p.num_docs_per_class:break
            full_text = row[0]
            full_text = full_text.replace("won't ", "will not ").replace("can't ", "can not ")
            full_text = ' '.join(work_tokenizer.tokenize(full_text))
            sentences = sentence_tokenizer.tokenize(full_text)

            preprocessed = []
            for sentence in sentences:
                # print(sentence)
                sentence = sentence.replace("'re ", "are ").replace("'ll ", "will ").replace("n't ", "not ").replace("'m ","am ").replace("'ve", "have")
                sentence = [word for word in normalizeString(sentence).split() if len(word) >= 2]
                sentence = [lemmatizer.lemmatize(word) for word in sentence.split(' ') if len(word) > 1]
                for word in sentence:
                    word_count[word] += 1
                # print(sentence)
                sentence = ' '.join(sentence)
                # eng.addSentence(sentence)
                preprocessed.append(sentence)
            # print(preprocessed)
            documents[class_label].append(deepcopy(preprocessed))


def data_to_tensor():
    # for label in ["great", "good", "typical"]:
    doc_list = []
    label_list = []
    for label in ["great", "good", "typical"]:
        lable_count = 0
        docs = documents[label]
        cur_label = class_dict[label]

        for doc in docs:
            # print(doc)
            # print(len(doc))
            lable_count += 1
            sentence_list = []
            for sentence in doc:
                indexes = indexes_from_sentence(sentence)
                sentence_list.append(deepcopy(indexes))
                # print(indexes)
            # a list of sentence indices
            doc_list.append(deepcopy(sentence_list))
            # print(doc_list)

        label_list += [cur_label] * lable_count
        # print(label_list)
        # a list of doc indices
    # doc_array = np.array(doc_list, dtype=object)

    result_dict = {"tokens": doc_list, "label": label_list}
    return json.dumps(result_dict)



def main():
    preprocess_and_construct_vocab("great")
    preprocess_and_construct_vocab("good")
    preprocess_and_construct_vocab("typical")
    # total = reduce((lambda x, y: x + y), (reduce((lambda x, y: x + y), documents.values())))
    [eng.addSentence(sent) for docs in documents.values() for doc in docs for sent in doc]

    # print(documents.values())
    # print(word_count)
    # print([(k,v) for (k,v) in word_count.items() if v > 5])
    # print(eng.word2index)
    # final_json = data_to_tensor()
    # print(final_json)
    # with open(p.out_filename + ".json", "w")as f:
    #     f.write(final_json)

main()
