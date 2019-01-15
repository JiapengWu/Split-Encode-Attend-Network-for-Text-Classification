import os
import unicodedata
import re
from multiprocessing import Pool
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.tokenize import stanford
import nltk
import sys


def construct_document_dataset():
    segment = 'document'
    data_test = pd.read_json(input_path + '{}_test.json'.format(dataset_name))
    data_train = pd.read_json(input_path + '{}_train.json'.format(dataset_name))
    global train_idx, val_idx
    train_idx, val_idx = train_test_split(data_train["id"], test_size=0.1, random_state=42)
    data_val = data_train.loc[data_train['id'].isin(val_idx)]
    data_train = data_train.loc[data_train['id'].isin(train_idx)]

    for (df, slice) in zip([data_train, data_val, data_test], ['train', 'val', 'test']):
        construct_word_dataset(df, slice, segment)
        construct_sentence_dataset(df, slice, segment)
    print("Word and Sentence document dataset construction finished")


def construct_paragraph_dataset():
    segment = 'paragraph'
    data_test = pd.read_json(input_path + '{}_paragraph_test.json'.format(dataset_name))
    data_train = pd.read_json(input_path + '{}_paragraph_train.json'.format(dataset_name))
    data_val = data_train.loc[data_train['id'].isin(val_idx)]
    data_train = data_train.loc[data_train['id'].isin(train_idx)]

    for (df, slice) in zip([data_train, data_val, data_test], ['train', 'val', 'test']):
        construct_word_dataset(df, slice, segment)
        construct_sentence_dataset(df, slice, segment)

    print("Word and Sentence paragraph dataset construction finished")


def construct_word_dataset(df, slice, segment):
    seg_path = '../project_data/{}/train_val_test/{}/word_sequence/'.format(dataset_name, segment)
    if not os.path.exists(seg_path):
        os.makedirs(seg_path)
    df['tokens'] = df['tokens'].apply(lambda x: x.split())
    file_name = "{}_{}_word.json".format(slice, segment)
    df.to_json(os.path.join(seg_path, file_name))
    print("Finished preprocessing {}.".format(file_name))


def construct_sentence_dataset(df, slice, segment):
    seg_path = '../project_data/{}/train_val_test/{}/sentence_split/'.format(dataset_name, segment)
    if not os.path.exists(seg_path):
        os.makedirs(seg_path)
    df['tokens'] = apply_by_multiprocessing(df['tokens'], sentence_tokenization, workers=20)
    file_name = "{}_{}_sentence.json".format(slice, segment)
    df.to_json(os.path.join(seg_path, file_name))
    print("Finished preprocessing {}.".format(file_name))


def sentence_tokenization(text):
    sentence_tokennized = sentence_tokenizer.tokenize(' '.join(text))
    return clean_sentences(sentence_tokennized)


def clean_sentences(text):
    result_text = []
    for line in text:
        line = line.strip()
        m = re.match(r'[^a-zA-z]+', line)
        if m and m.end() == len(line):  # the whole sentence is a match
            continue
        else:
            line = re.sub(r"[0-9]+", "0", line)
            result_text.append(line)
    return result_text


def _apply_df(args):
    df, func, kwargs = args
    return df.apply(func, **kwargs)


def apply_by_multiprocessing(df, func, **kwargs):
    workers = kwargs.pop('workers')
    pool = Pool(processes=workers)
    result = pool.map(_apply_df, [(d, func, kwargs)
            for d in np.array_split(df, workers)])
    pool.close()
    return pd.concat(list(result))


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    s = re.sub(r"\\n", '\n', s)
    s = re.sub(r"\n{1,}", '\n', s)
    if first: s = re.sub(r"\n", ' (p) ', s)
    s = re.sub(r'\\"', '\"', s)
    s = re.sub(r"[^A-Za-z0-9(),!?\'\":\-]", " ", s)
    s = re.sub(r"\'s", " \'s", s)
    s = re.sub(r"\'ve", " \'ve", s)
    s = re.sub(r"n\'t", " n\'t", s)
    s = re.sub(r"\'re", " \'re", s)
    s = re.sub(r"\'d", " \'d", s)
    s = re.sub(r"\'ll", " \'ll", s)
    s = re.sub(r",", " , ", s)
    s = re.sub(r"!", " ! ", s)
    s = re.sub(r"\-", " - ", s)
    s = re.sub(r":", " : ", s)
    s = re.sub(r"\(", " ( ", s)
    s = re.sub(r"\)", " ) ", s)
    s = re.sub(r"\?", " ? ", s)
    s = re.sub(r"<br />", " ", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = unicodeToAscii(s.lower().strip())
    return s


def preprocess_dataset():
    data_test = pd.read_csv(input_path + 'test.csv', header=None)
    data_train = pd.read_csv(input_path + 'train.csv', header=None)

    test_length = len(data_test[1])
    train_length = len(data_train[1])

    test_idx = list(range(train_length, train_length + test_length))
    train_idx = list(range(train_length))

    for df, idx, slice in zip([data_test, data_train], [test_idx, train_idx], ["test", "train"]):
        print("finished loading %d data instances" % len(df))
        if dataset_name == 'ag_news':
            df[1] = df.apply(lambda row: row[1] + " " + row[2], axis=1)
            df = df[[0, 1]]

        df.columns = ["label", "tokens"]
        df['id'] = pd.Series(idx)
        df['tokens'] = apply_by_multiprocessing(df['tokens'], normalizeString, workers=20)
        print("finished preprocessing %d data instances" % len(df['label']))

        df.to_json(input_path + "{}_{}.json".format(dataset_name, slice))

    print("Finished processing document dataset")


def preprocess_paragraph_dataset():
    def split_paragraphs(row):
        s = row['tokens']
        return s.split('( p )')

    data_test = pd.read_json(input_path + '{}_test.json'.format(dataset_name))
    data_train = pd.read_json(input_path + '{}_train.json'.format(dataset_name))

    for df, slice in zip([data_test, data_train], ["test", "train"]):
        idx = []
        label_list = []
        token_list = []
        for index, row in df.iterrows():
            id, label = (row['id'], row['label'])
            paragraphs = split_paragraphs(row)
            # if len(paragraphs) != 1 : print(len(paragraphs))
            list(map(lambda x: token_list.append(x), paragraphs))
            idx += [id] * len(paragraphs)
            label_list += [label] * len(paragraphs)
        result = pd.concat([pd.Series(idx), pd.Series(label_list), pd.Series(token_list)], axis=1)
        result.columns = ['id', 'label', 'tokens']
        result.to_json(os.path.join(input_path, "{}_paragraph_{}.json".format(dataset_name, slice)))

    print("Finished constructing paragraph dataset")


def get_fixed_length_paragraph(tokens, length):
    total_length = len(tokens)
    if total_length <= length:
        return [tokens]
    paragraphs = []
    paragraph = []
    for (i, sent) in enumerate(tokens, 1):
        condition = i % length == 0 or i == total_length
        paragraph.append(sent)
        if condition:
            paragraphs.append(paragraph)
            paragraph = []
    # print(tokens)
    # print(paragraphs)
    return paragraphs


def construct_fixed_length_dataset(data_dir):
    input_path = '../project_data/{}/train_val_test/document/sentence_split/'.format(data_dir)
    output_path = '../project_data/{}/train_val_test/fixed_length/sentence_split/'.format(data_dir)
    word_seq_output_path = '../project_data/{}/train_val_test/fixed_length/word_sequence/'.format(data_dir)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(word_seq_output_path):
        os.makedirs(word_seq_output_path)

    for slice in ['train', 'val', 'test']:
        fname = "{}_document_sentence.json".format(slice)
        df = pd.read_json(os.path.join(input_path, fname))
        idx = []
        label_list = []
        token_list = []
        for index, row in df.iterrows():
            id, label, tokens = (row['id'], row['label'], row['tokens'])

            paragraphs = get_fixed_length_paragraph(tokens, 3)
            list(map(lambda x: token_list.append(x), paragraphs))
            idx += [id] * len(paragraphs)
            label_list += [label] * len(paragraphs)

        result = pd.concat([pd.Series(idx), pd.Series(label_list), pd.Series(token_list)], axis=1)
        result.columns = ['id', 'label', 'tokens']
        result.to_json(os.path.join(output_path, "{}_paragraph_sentence.json".format(slice)))

        result['tokens'] = result['tokens'].apply(lambda x: " ".join(x).split())
        result.to_json(os.path.join(word_seq_output_path, "{}_paragraph_word.json".format(slice)))

    print("Fixed-length paragraph dataset construction finished")

# def convert_sent_to_word_seq(sent_list):
#     for sents

if __name__ == '__main__':
    # dataset_name = sys.argv[1]
    # sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    # work_tokenizer = stanford.StanfordTokenizer(
    #     path_to_jar="../stanford-postagger-2018-02-27/stanford-postagger-3.9.1.jar")
    # input_path = '../project_data/{}/'.format(dataset_name)
    #
    # first = True
    # preprocess_dataset()
    # preprocess_paragraph_dataset()
    # first = False
    # preprocess_dataset()
    #
    # construct_document_dataset()
    # construct_paragraph_dataset()

    construct_fixed_length_dataset('linguistic_quality')
    construct_fixed_length_dataset('yelp')
