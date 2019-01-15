import pandas as pd
import numpy as np
from os import path
from utility import *
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt


def load_document_data():
    dir = '../project_data/sentiment/train_val_test/document/sentence_split'
    train = pd.read_json(path.join(dir, "train_document_sentence.json"))
    val = pd.read_json(path.join(dir, "val_document_sentence.json"))
    test = pd.read_json(path.join(dir, "test_document_sentence.json"))
    data = pd.concat([train, val, test], axis=0)
    return train, val, test, data


def load_paragraph_data():
    dir = '../project_data/sentiment/train_val_test/paragraph/sentence_split'
    train = pd.read_json(path.join(dir, "train_paragraph_sentence.json"))
    val = pd.read_json(path.join(dir, "val_paragraph_sentence.json"))
    test = pd.read_json(path.join(dir, "test_paragraph_sentence.json"))
    data = pd.concat([train, val, test], axis=0)
    return train, val, test, data


def calculate_sent(tokens):
    sent_num_list = tokens.apply(lambda x: len(x))
    print("Average number of sentences: {}".format(np.mean(sent_num_list.values)))
    print("Max number of sentences: {}".format(np.max(sent_num_list.values)))
    return sent_num_list


def calculate_words(tokens):
    def sum_words(sents):
        return np.sum([len(sent.split()) for sent in sents])

    word_num_list = tokens.apply(lambda x: sum_words(x))
    print("Average number of words: {}".format(np.mean(word_num_list.values)))
    print("Max number of words: {}".format(np.max(word_num_list.values)))
    return word_num_list


def calculate_paragraphs(tokens):
    paragraph_counts = tokens.groupby("id").agg('count')
    print("Average number of paragraphs: {}".format(np.mean(paragraph_counts.values)))
    print("Max number of paragraphs: {}".format(np.max(paragraph_counts.values)))
    return paragraph_counts['tokens']


def document_dataset_statistics():
    train, val, test, data = load_document_data()
    neg = data[data['label'] == 0][['id', 'label', 'tokens']]
    pos = data[data['label'] == 1][['id', 'label', 'tokens']]
    print("Number of documents: {}, {}".format(len(neg), len(pos)))
    print("Total: ")
    plot_histogram(calculate_sent(data['tokens']), fname='imdb sentence statistics')
    plot_histogram(calculate_words(data['tokens']), fname='imdb word statistics')

    plot_histogram(calculate_sent(val['tokens']), fname='imdb 10% val set sentence statistics')
    plot_histogram(calculate_words(val['tokens']), fname='imdb 10% val set word statistics')

    plot_histogram(calculate_sent(test['tokens']), fname='imdb 10% test set sentence statistics')
    plot_histogram(calculate_words(test['tokens']), fname='imdb 10% test set word statistics')

    # print("Number of documents: {}, {}".format(len(neg), len(pos)))
    # print("Neg: ")
    # calculate_sent(neg['tokens'])
    # calculate_words(neg['tokens'])
    # print("Pos:")
    # calculate_sent(pos['tokens'])
    # calculate_words(pos['tokens'])


def paragraph_dataset_statistics():
    train, val, test, data = load_paragraph_data()
    paragraph_counts = calculate_paragraphs(data[['id', 'tokens']])
    plot_histogram(paragraph_counts, fname='imdb paragraph statistics')

    paragraph_counts = calculate_paragraphs(val[['id', 'tokens']])
    plot_histogram(paragraph_counts, fname='imdb 10% val set paragraph statistics')

    paragraph_counts = calculate_paragraphs(test[['id', 'tokens']])
    plot_histogram(paragraph_counts, fname='imdb 10% test set paragraph statistics')


def plot_histogram(x, xlabel=None, ylabel="Counts", title=None, fname="tmp"):
    max_x = max(x);min_x = min(x)
    n, bins, patches = plt.hist(x, 50, facecolor='g', alpha=0.75)
    if xlabel:
        plt.xlabel("Number of {}".format(xlabel))
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.axis([min_x, max_x, 0, 7/6 * max(n)])
    plt.grid(True)
    plt.savefig('plots/{}.png'.format(fname))
    plt.clf()


if __name__ == '__main__':
    document_dataset_statistics()
    paragraph_dataset_statistics()