import pandas as pd
import parameters as p
import numpy as np
from sklearn.model_selection import train_test_split
from xml.etree import ElementTree as ET
import ast


def construct_full_json():
    data_train = pd.read_json(p.train_test_val_data_dir + "train.json")
    data_val = pd.read_json(p.train_test_val_data_dir + "val.json")
    data_test = pd.read_json(p.train_test_val_data_dir + "test.json")
    full = pd.concat([data_train, data_val, data_test], axis=0)
    full = full[['id', 'label', 'tokens']]
    full.to_csv(p.train_test_val_data_dir + "full.csv")


def filter_verygood_typical():
    data = pd.read_csv(p.train_test_val_data_dir + "full.csv")
    typical = data[data['label'] == 0][['id', 'label', 'tokens']]
    very_good = data[data['label'] == 1][['id', 'label', 'tokens']]
    great = data[data['label'] == 2][['id', 'label', 'tokens']]
    typical.to_csv(p.train_test_val_data_dir + "typical.csv")
    very_good.to_csv(p.train_test_val_data_dir + "very_good.csv")
    great.to_csv(p.train_test_val_data_dir + "great.csv")


def repartition_dataset():
    typical = pd.read_csv(p.train_test_val_data_dir + "typical.csv")
    very_good = pd.read_csv(p.train_test_val_data_dir + "very_good.csv")
    great = pd.read_csv(p.train_test_val_data_dir + "great.csv")
    great['label'] = np.array([1] * len(great))
    very_good = pd.concat([great, very_good], axis=0)

    print(very_good.shape)
    print(typical.shape)
    df = pd.concat([typical, very_good], axis=0)
    df = df[['id', 'label', 'tokens']]
    print(df)
    data_train, data_test = train_test_split(df, test_size=p.text_size, random_state=42)
    data_train, data_val = train_test_split(data_train, test_size=p.val_size, random_state=41)
    data_train.to_csv(p.train_test_val_data_dir + "train.csv")
    data_val.to_csv(p.train_test_val_data_dir + "val.csv")
    data_test.to_csv(p.train_test_val_data_dir + "test.csv")


def check_class_size():
    train = pd.read_csv(p.train_test_val_data_dir + "train.csv")
    # val = pd.read_csv(p.train_test_val_data_dir + "val.csv")
    # test = pd.read_csv(p.train_test_val_data_dir + "test.csv")
    great = train[train['label'] == 0][['id', 'label', 'tokens']]
    typical = train[train['label'] == 1][['id', 'label', 'tokens']]
    print(great.shape, typical.shape)


def check_vocab_size():
    dictionary = pd.read_pickle(p.dict_path)
    print(len(dictionary.word2idx.keys()))
    print(dictionary.word2idx.values())


def check_label_size():
    data_train = pd.read_csv(p.train_test_val_data_dir + "train.csv")
    y_train = data_train.label.values
    print(set(y_train))


def augment_dataset():
    train = pd.read_csv(p.sent_split_dir + "train_multi_label.csv")
    typical = train[train['label'] == 0][['id', 'label', 'multi_label', 'tokens']]
    great = train[train['label'] == 1][['id', 'label', 'multi_label', 'tokens']]
    frac = great.sample(2108)
    replica = pd.concat([great] * 4, ignore_index=True)
    train = pd.concat([replica, frac, typical])
    train = train.sample(frac=1, random_state=42)
    train.to_csv(p.sent_split_dir + "train_multi_label_augmented.csv")
    # print(train)


def is_topic(topic, article_path):
    e = ET.parse(article_path).getroot()
    for category in e.iter("classifier"):
        if isinstance(category.text, str):
            topic_list = category.text.split("/")
            for word in topic_list:
                if word == topic:
                    return True

    return False


def find_labels(row, topics):
    dir = './typical/' if int(row['label']) == 0 else './good/'
    id = str(row['id']) + '.xml'
    topic_vec = [0] * 11
    try:
        f = open(dir + id, 'r')
        article_path = dir + id
        f.close()
    except:
        article_path = './great/' + id
    for topic in topics.keys():
        if is_topic(topic, article_path):
            idx = topics[topic]
            topic_vec[idx] = 1
    # return row['id'], row['label'], row['tokens'], np.array(topic_vec)
    return topic_vec


def split_words(doc):
    doc = ast.literal_eval(doc)
    doc = ' '.join(doc)
    return doc.split()


def construct_multi_label_dataset(filename):
    topics = {'Medicine and Health': 0, 'Research': 1, "Space": 2, 'Physics': 3, 'Computers and the Internet': 4
              , 'Brain': 5, 'Evolution': 6, 'Disasters': 7, 'Religion and Churches': 8, 'Language and Languages': 9
              , 'Environment': 10}
    df = pd.read_csv(p.train_test_val_data_dir + filename + ".csv")
    print(df.shape)
    df = df[['id', 'label', 'tokens']]
    df['multi_label'] = df.apply(lambda row: find_labels(row, topics), axis=1)
    df.to_csv(p.train_test_val_data_dir + filename + '_multi_label.csv')
    print(df.shape)
    df['tokens'] = df['tokens'].apply(lambda doc: split_words(doc))
    df.to_csv(p.train_test_val_data_dir + filename + '_single_word_multi_label.csv')
    print(df.shape)


def squeeze_sents(paragraph):
    nested = list(map(lambda x: x.split(), paragraph))
    return [word for sent in nested for word in sent]


def construct_paragraph_single_word(filename):
    print(p.sent_split_dir + filename + "_paragraph.json")
    df = pd.read_json(p.sent_split_dir + filename + "_paragraph.json")
    df['tokens'] = df['tokens'].apply(lambda paragraph: squeeze_sents(paragraph))
    df.to_json(p.word_seq_dir + filename + "_paragraph_single_word.json")


def construct_3_layer_dataset():

    data_train = pd.read_json(p.sent_split_dir + "train_clustered_sent_split.json")
    data_val = pd.read_json(p.sent_split_dir + "val_clustered_sent_split.json")
    data_test = pd.read_json(p.sent_split_dir + "test_clustered_sent_split.json")
    zip_file = zip([data_train, data_val, data_test], ['train', 'val', 'test'])
    for file, fname in zip_file:
        print("%s" % fname)
        grouped = file.groupby("id")
        idx = []
        labels = []
        tokens = []
        for name, group in grouped:
            # print(name)
            # print(group)
            # print(group.tokens.values.tolist())
            idx.append(name)
            labels.append(group.label.values[0])
            tokens.append(group.tokens.values.tolist())
        result = pd.concat([pd.Series(idx), pd.Series(labels), pd.Series(tokens)], axis=1)
        result.columns = ['id', 'label', 'tokens']
        result.to_json(p.paragraph_split_dir + fname + '_paragraph_split.json')


def construct_start_mid_end():
    from math import ceil, floor
    import itertools
    import os
    data_train = pd.read_json(p.sent_split_dir + "train_paragraph_ordered.json")
    data_val = pd.read_json(p.sent_split_dir + "val_paragraph_ordered.json")
    data_test = pd.read_json(p.sent_split_dir + "test_paragraph_ordered.json")
    zip_file = zip([data_train, data_val, data_test], ['train', 'val', 'test'])

    for file, fname in zip_file:
        print("%s" % fname)
        grouped = file.groupby("id")
        idx = []
        label_list = []
        token_list = []
        for name, group in grouped:
            # print(group)
            label = group.label.values[0]
            tokens = group.tokens.tolist()
            first = tokens[:ceil(0.2 * len(tokens))]
            mid = tokens[ceil(0.2 * len(tokens)): floor(0.8 * len(tokens))]
            end = tokens[floor(0.8 * len(tokens)):]
            for cluster in [first, mid, end]:
                idx.append(name)
                label_list.append(label)
                token_list.append(list(itertools.chain.from_iterable(cluster)))
        result = pd.concat([pd.Series(idx), pd.Series(label_list), pd.Series(token_list)], axis=1)
        result.columns = ['id', 'label', 'tokens']
        # print(result)
        path = p.sent_split_dir + '3 parts/'
        if not os.path.exists(path):
            os.makedirs(path)
        result.to_json(path + fname + '_3_parts_split.json')


def find_author(row, aimed_authors):

    dir = 'typical/' if int(row['label']) == 0 else 'good/'
    id = str(row['id']) + '.xml'
    try:
        f = open('../project_data/nyt/' + dir + id, 'r')
        article_path = dir + id
        f.close()
    except:
        article_path = 'great/' + id
    e = ET.parse('../project_data/nyt/' + article_path).getroot()
    try:
        author = e.find('./body/body.head/byline[2]').text
        if author not in aimed_authors:
            return '-1'
        # author = e.find('./body/body.head/byline[@class="print_byline"]').text
    except:
        return '-1'
        # author = 'Undecided'
    return author


def construct_auth_dict():
    from collections import Counter
    df = pd.read_csv('../project_data/preprocessed/full.csv')
    authors_list = df.apply(lambda row: find_author(row), axis=1).tolist()
    idx_auther_dict = dict(zip(df['id'].tolist(), authors_list))
    print(Counter(authors_list))
    author_index_dict = {}
    for idx, author in enumerate(list(set(authors_list))):
        author_index_dict[author] = idx
    return idx_auther_dict, author_index_dict


def construct_authorship_dataset():
    import os
    path = '../project_data/authorship/'
    if not os.path.exists(path):
        os.makedirs(path)

    aimed_authors = [
        'Fountain, Henry', 'Kolata, Gina',
        'Altman, Lawrence K', 'Wade, Nicholas',
        'Pollack, Andrew', 'Grady, Denise',
        'Brody, Jane E', 'Chang, Kenneth',
        'Markoff, John', 'Wilford, John Noble'
    ]

    # df = pd.read_csv('../project_data/preprocessed/full.csv')

    df_train = pd.read_csv('../project_data/train_val_test/paragraph/original/train.csv', header=None)
    df_val = pd.read_csv('../project_data/train_val_test/paragraph/original/val.csv', header=None)
    df_test = pd.read_csv('../project_data/train_val_test/paragraph/original/test.csv', header=None)
    df = pd.concat([df_train, df_val, df_test], axis=0)
    df.columns = ['id', 'label', 'tokens']
    # df_good = pd.read_csv('../project_data/good_great_typ/good.csv', header=None)
    # df_typ = pd.read_csv('../project_data/good_great_typ/typical.csv', header=None)
    # df_great = pd.read_csv('../project_data/good_great_typ/great.csv', header=None)
    # df = pd.concat([df_good, df_typ, df_great], axis=0)
    # df.columns = ['id', 'tokens', 'label']
    # print(df)
    df['label'] = df.apply(lambda row: find_author(row, aimed_authors), axis=1)
    # filter data based on authors
    df = df[df['label'] != '-1']

    print(df['label'])
    author_index_dict = {}
    for idx, author in enumerate(aimed_authors):
        author_index_dict[author] = idx

    df['label'] = df.apply(lambda row: author_index_dict[row['label']], axis=1)
    df.to_csv(path + "paragraph.csv")


def construct_word_seq_from_sent_seq():
    import os
    path = '../project_data/authorship/train_val_test/paragraph/word_sequence/'
    if not os.path.exists(path):
        os.makedirs(path)
    for fname in ['train', 'val', 'test']:
        df = pd.read_json('../project_data/authorship/train_val_test/paragraph/sentence_split/' + fname + '_paragraph.json')
        df['tokens'] = df['tokens'].apply(lambda row: ' '.join(row).split())
        df.to_json(path + '{}_single_word.json'.format(fname))


if __name__ == '__main__':
    # construct_start_mid_end()
    # construct_authorship_dataset()
    construct_word_seq_from_sent_seq()