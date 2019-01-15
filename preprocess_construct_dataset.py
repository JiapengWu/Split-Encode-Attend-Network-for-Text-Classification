import os
import unicodedata
import re
from multiprocessing import Pool
import pandas as pd
import parameters as p
import numpy as np
from sklearn.model_selection import train_test_split
from copy import deepcopy
import nltk


def construct_imdb_dataset():
    input_path = '../project_data/sentiment/'
    segment = 'document'

    data_test = pd.read_json(input_path + 'imdb_test.json')
    data_train = pd.read_json(input_path + 'imdb_train.json')
    if original_split:
        data_train, data_val = train_test_split(data_train, test_size=0.1, random_state=42)
    else:
        df = pd.concat([data_train, data_test], axis=0)
        data_train, data_val, data_test = np.split(df.sample(frac=1), [int(.6 * len(df)), int(.8 * len(df))])

    for (df, slice) in zip([data_train, data_val, data_test], ['train', 'val', 'test']):
        construct_word_dataset(df, slice, segment)
        construct_sentence_dataset(df, slice, segment)
    print("Word and Sentence dataset construction finished")


def construct_word_dataset(df, slice, segment):
    seg_path = '../project_data/sentiment/{}/train_val_test/{}/word_sequence/'.format(dir_name, segment)
    if not os.path.exists(seg_path):
        os.makedirs(seg_path)
    df['tokens'] = df['tokens'].apply(lambda x: x.split())
    file_name = "{}_{}_word.json".format(slice, segment)
    df = df.reset_index()[['id', 'tokens', 'label']]
    df.to_json(os.path.join(seg_path, file_name))
    print("Finished preprocessing {}.".format(file_name))


def construct_sentence_dataset(df, slice, segment):
    seg_path = '../project_data/sentiment/{}/train_val_test/{}/sentence_split/'.format(dir_name, segment)
    if not os.path.exists(seg_path):
        os.makedirs(seg_path)
    df['tokens'] = apply_by_multiprocessing(df['tokens'], sentence_tokenization, workers=20)
    file_name = "{}_{}_sentence.json".format(slice, segment)
    df = df.reset_index()[['id', 'tokens', 'label']]
    df.to_json(os.path.join(seg_path, file_name))
    print("Finished preprocessing {}.".format(file_name))


sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
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


def extract_sentences(id, tokens, outpath):
    fname = "{}.txt".format(id)
    text = "\n".join(tokens)
    with open(os.path.join(outpath, fname), "w") as f:
        f.write(text)


def restore_txt_files():
    input_path = '../project_data/sentiment/train_val_test/document/sentence_split/'
    outpath = '../GraphSeg/data/seg-input/'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    for slice in ['train', 'val', 'test']:
        fname = "{}_document_sentence.json".format(slice)
        df = pd.read_json(os.path.join(input_path, fname))
        df.apply(lambda row: extract_sentences(row['id'], row['tokens'], outpath), axis=1)


def get_paragraphs(id):
    text_path = '../GraphSeg/data/seg-output/'
    with open(os.path.join(text_path, "{}.txt".format(id))) as f:
        lines = f.readlines()
        paragraphs = []
        paragraph = []
        for line in lines:
            line = line.strip()
            if '==========' in line:
                paragraphs.append(deepcopy(paragraph))
                paragraph = []
            else:
                m = re.match(r'[^a-zA-z]+', line)
                if m and m.end() == len(line): # the whole sentence is a match
                    continue
                else:
                    line = re.sub(r"[0-9]+", "0", line)
                    paragraph.append(line)

        return paragraphs


def construct_paragraph_sent_dataset():
    # id, tokens, label,
    input_path = '../project_data/sentiment/{}/train_val_test/document/sentence_split/'.format(dir_name)
    output_path = '../project_data/sentiment/{}/train_val_test/paragraph/sentence_split/'.format(dir_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for slice in ['train', 'val', 'test']:
        fname = "{}_document_sentence.json".format(slice)
        df = pd.read_json(os.path.join(input_path, fname))
        idx = []
        label_list = []
        token_list = []
        for index, row in df.iterrows():
            id, label = (row['id'], row['label'])

            if id == 3118 or id == 3119:
                paragraphs = [['smallville episode justice is the best episode of smallville !', 'its my favorite episode of smallville !']]
            elif id == 7823:
                paragraphs = [['this movie is the only movie to feature a scene in which michael jackson wields a tommy gun .', 'plain and simple .', 'this movie rocks because it is freaking hilarious !',
                               'it may be creepy to see jacko w little kids but this movie also stars .......................................... wait for it ..................... joe pesci !',
                               'think about it joe pesci and jacko with tommy guns throwing coins into jukeboxes from 20 feet away ?', 'what s not to like ?', 'as stated before this movie rocks !']]
            else:
                paragraphs = get_paragraphs(id)
            list(map(lambda x: token_list.append(x), paragraphs))
            idx += [id] * len(paragraphs)
            label_list += [label] * len(paragraphs)

        result = pd.concat([pd.Series(idx), pd.Series(label_list), pd.Series(token_list)], axis=1)
        result.columns = ['id', 'label', 'tokens']
        result.to_json(os.path.join(output_path, "{}_paragraph_sentence.json".format(slice)))
    print("Paragraph dataset construction finished")


def construct_paragraph_word_dataset():
    # for seg in "fixed_length", "paragraph":
        dir_name = "linguistic_quality"
        seg = "paragraph"
        input_path = '../project_data/{}/train_val_test/{}/sentence_split/'.format(dir_name, seg)
        output_path = '../project_data/{}/train_val_test/{}/word_sequence/'.format(dir_name, seg)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for slice in ['train', 'val', 'test']:
            fname = "{}_paragraph_sentence.json".format(slice)
            print(os.path.join(input_path, fname))
            df = pd.read_json(os.path.join(input_path, fname))
            df['tokens'] = df['tokens'].apply(lambda x: " ".join(x).split())
            df.to_json(os.path.join(output_path, "{}_paragraph_word.json".format(slice)))


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
    print(tokens)
    print(paragraphs)
    return paragraphs


def construct_fixed_length_dataset():
    input_path = '../project_data/sentiment/{}/train_val_test/document/sentence_split/'.format(dir_name)
    output_path = '../project_data/sentiment/{}/train_val_test/fixed_length/sentence_split/'.format(dir_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for slice in ['train', 'val', 'test']:
        fname = "{}_document_sentence.json".format(slice)
        df = pd.read_json(os.path.join(input_path, fname))
        idx = []
        label_list = []
        token_list = []
        for index, row in df.iterrows():
            id, label, tokens = (row['id'], row['label'], row['tokens'])

            if id == 3118 or id == 3119:
                paragraphs = [['smallville episode justice is the best episode of smallville !', 'its my favorite episode of smallville !']]
            elif id == 7823:
                paragraphs = [['this movie is the only movie to feature a scene in which michael jackson wields a tommy gun .', 'plain and simple .', 'this movie rocks because it is freaking hilarious !',
                               'it may be creepy to see jacko w little kids but this movie also stars .......................................... wait for it ..................... joe pesci !',
                               'think about it joe pesci and jacko with tommy guns throwing coins into jukeboxes from 20 feet away ?', 'what s not to like ?', 'as stated before this movie rocks !']]
            else:
                paragraphs = get_fixed_length_paragraph(tokens, 3)
            list(map(lambda x: token_list.append(x), paragraphs))
            idx += [id] * len(paragraphs)
            label_list += [label] * len(paragraphs)

        result = pd.concat([pd.Series(idx), pd.Series(label_list), pd.Series(token_list)], axis=1)
        result.columns = ['id', 'label', 'tokens']
        result.to_json(os.path.join(output_path, "{}_paragraph_sentence.json".format(slice)))
    print("Fixed-length paragraph dataset construction finished")


def preprocess_ling_dataset():
    df = pd.read_csv(os.path.join(p.data_dir, 'full.csv'))
    print("finished loading %d data instances" % len(df))
    df_tokens = df['tokens']
    df_tokens = apply_by_multiprocessing(df_tokens, sentence_tokenization, workers=20)

    df = pd.DataFrame(list(zip(df['id'], df_tokens, df['label'])))
    df.columns = ['id', 'tokens', 'label']
    data_train, data_test = train_test_split(df, test_size=p.text_size, random_state=42)
    data_train, data_val = train_test_split(data_train, test_size=p.val_size, random_state=41)

    print("finished preprocessing %d data instances" % len(df['label']))

    output_dir = p.data_dir + 'train_val_test/document/sentence_split/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data_train.to_json(output_dir + "train_document_sent.json")
    data_val.to_json(output_dir + "val_document_sent.json")
    data_test.to_json(output_dir + "test_document_sent.json")


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


def preprocess_sentences(text):
    from nltk.tokenize import stanford
    import nltk
    sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    work_tokenizer = stanford.StanfordTokenizer(
        path_to_jar="../stanford-postagger-2018-02-27/stanford-postagger-3.9.1.jar")

    def unicodeToAscii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    def normalizeString(s):
        s = re.sub(r"[^A-Za-z0-9(),!?\'\"-:]", " ", s)
        s = re.sub(r"\'s", " \'s", s)
        s = re.sub(r"\'ve", " \'ve", s)
        s = re.sub(r"n\'t", " n\'t", s)
        s = re.sub(r"\'re", " \'re", s)
        s = re.sub(r"\'d", " \'d", s)
        s = re.sub(r"\'ll", " \'ll", s)
        s = re.sub(r",", " , ", s)
        s = re.sub(r"!", " ! ", s)
        s = re.sub(r"\(", " \( ", s)
        s = re.sub(r"\)", " \) ", s)
        s = re.sub(r"\?", " \? ", s)
        s = re.sub(r"<br />", " ", s)
        s = re.sub(r"\s{2,}", " ", s)
        s = unicodeToAscii(s.lower().strip())
        return s

    text = ' '.join(work_tokenizer.tokenize(text))
    sentences = sentence_tokenizer.tokenize(text)

    preprocessed = []
    for sentence in sentences:
        sentence = [word for word in normalizeString(sentence).split() if len(word) >= 2]
        sentence = ' '.join(sentence)
        if sentence:
            preprocessed.append(sentence)
    return preprocessed



if __name__ == '__main__':
    # original_split = False
    # dir_name = '4-1-5 split' if original_split else '8-1-1 split'
    # construct_imdb_dataset()
    # construct_paragraph_sent_dataset()
    # construct_fixed_length_dataset()
    # construct_paragraph_word_dataset()
    construct_paragraph_word_dataset()