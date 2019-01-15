import pandas as pd
import numpy as np
import sys



def calculate_sents():
    sent_num_list = data.apply(lambda x: len(x))
    print("Average number of sentences: {}".format(np.mean(sent_num_list.values)))
    print("Max number of sentences: {}".format(np.max(sent_num_list.values)))


def calculate_words():
    def sum_words(sents):
        return np.sum([len(sent.split()) for sent in sents])

    word_num_list = data.apply(lambda x: sum_words(x))
    print("Average number of words: {}".format(np.mean(word_num_list.values)))
    print("Max number of words: {}".format(np.max(word_num_list.values)))


def calculate_paras():
    para_num_list = []
    for grouped in train_grouped, val_grouped, test_grouped:
        for name, group in grouped:
            # print(group)
            para_num_list.append(len(group))
    print("Average number of paragraphs: {}".format(np.mean(para_num_list)))
    print("Max number of paragraphs: {}".format(np.max(para_num_list)))


if __name__ == '__main__':
    dataset_name = sys.argv[1]
    if dataset_name == 'ling':
        dataset_name = 'linguistic_quality'
    elif dataset_name == 'sent':
        dataset_name = 'sentiment/4-1-5 split'
    para_dir = '../project_data/{}/train_val_test/paragraph/sentence_split/'.format(dataset_name)
    doc_dir = '../project_data/{}/train_val_test/document/sentence_split/'.format(dataset_name)

    train = pd.read_json(para_dir + "train_paragraph_sentence.json")
    val = pd.read_json(para_dir + "val_paragraph_sentence.json")
    test = pd.read_json(para_dir + "test_paragraph_sentence.json")

    train_document = pd.read_json(doc_dir + "train_document_sentence.json")
    val_document = pd.read_json(doc_dir + "val_document_sentence.json")
    test_document = pd.read_json(doc_dir + "test_document_sentence.json")

    data = pd.concat([train_document.tokens, val_document.tokens, test_document.tokens], axis=0)

    train_grouped = train.groupby("id")
    val_grouped = val.groupby("id")
    test_grouped = test.groupby("id")

    calculate_paras()
    calculate_words()
    calculate_sents()