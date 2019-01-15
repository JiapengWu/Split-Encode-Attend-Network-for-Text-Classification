from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import parameters as p
import pandas as pd
from dataset import TextDataset
from utility import *
import matplotlib
matplotlib.use('agg')
from word_seq_models import *
import os


def load_val_test_data():
    data_val = pd.read_json(p.word_seq_dir + "val_paragraph_word.json")
    data_test = pd.read_json(p.word_seq_dir + "test_paragraph_word.json")
    if config.dataset == "yelp":
        data_val['label'] = data_val['label'].apply(lambda x: 0 if x == 1 else 1).astype('int64')
        data_test['label'] = data_test['label'].apply(lambda x: 0 if x == 1 else 1).astype('int64')
    return data_val, data_test


def get_prediction(grouped, combine):
    total_pred = list()
    total_targets = list()
    #
    # id_prob = dict()
    # id_pred = dict()
    # id_label = dict()
    word_model.eval()

    for name, group in grouped:
        tokens = TextDataset._text2idx(group.tokens, word2idx)
        labels = np.array(group.label.values)
        tokens, labels = process_batch(tokens, labels)
        # prediction, soft_preds = test_accuracy_full_batch(tokens, labels, word_attn, sent_attn)
        logits, hidden = word_model.forward(tokens)
        soft_preds = F.softmax(logits, dim=1)
        _, prediction = torch.max(soft_preds, 1)
        if combine == 'majority':
            document_pred = 1 if torch.nonzero(prediction).size(0) >= prediction.shape[0] / 2 else 0
        elif combine == 'avg':
            document_pred = 1 if torch.mean(soft_preds, 0)[0].item() < 0.5 else 0
        target = labels[0].item()


        total_pred.append(document_pred)
        total_targets.append(target)

    return total_pred, total_targets


def evaluate(combine='avg'):
    # data_train = pd.read_json(p.sent_split_dir + "val_clustered_sent_split.json")

    """evaluate the model """
    total_pred, total_targets = \
        get_prediction(val_grouped, combine)
    total_pred = np.array(total_pred)
    total_targets = np.array(total_targets)
    precision, recall, f1, _ = precision_recall_fscore_support(total_targets, total_pred)
    acc = accuracy_score(total_targets, total_pred)
    conf_matrix = confusion_matrix(total_targets, total_pred)
    print(combine + " | val set Precision: " + str(precision) + " | Recall: " + str(recall) + " | F1-score: " + str(f1) +
          " | Accuracy: " + str(acc))
    print("Confusion matrix: \n" + str(conf_matrix))


    total_pred, total_targets = \
        get_prediction(test_grouped, combine)
    total_pred = np.array(total_pred)
    total_targets = np.array(total_targets)
    precision, recall, f1, _ = precision_recall_fscore_support(total_targets, total_pred)
    acc = accuracy_score(total_targets, total_pred)
    conf_matrix = confusion_matrix(total_targets, total_pred)
    print(combine + " |test set Precision: " + str(precision) + " | Recall: " + str(recall) + " | F1-score: " + str(f1) +
          " | Accuracy: " + str(acc))
    print("Confusion matrix: \n" + str(conf_matrix))


def process_batch(tokens, labels):
    tokens = pad_2d_batch(tokens)
    return tokens.cuda(), Variable(torch.from_numpy(labels), requires_grad=False).cuda()


def load_model(config, word2idx):
    if word_model_name == 'cnn':
        word_model = CNN_Text(config, word2idx)
    elif word_model_name.lower() == 'lstm':
        word_model = RNNPooling(config, word2idx)
    else:
        raise Exception("Please input a proper word model name!")
    if config.model_exp_num == 151:
        index = 13
    else:
        if os.path.isfile('./log_exp_{}'.format(config.model_exp_num)):
            log_file = './log_exp_{}'.format(config.model_exp_num)
        else:
            log_file = 'logs/log_exp_{}'.format(config.model_exp_num)
        val_set = []
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'val set' in line:
                    val_set.append(line)

        f1s = [float(line[-6:]) for line in val_set]
        index = np.argmax(f1s)

    print("loading model {} at number {} epoch".format(config.model_exp_num, index + 1))
    pre_cnn = torch.load(
        "./experiments/{}/models/word_model.epoch-{:02d}.pt".format(config.model_exp_num, index))

    word_model.load_state_dict(pre_cnn)

    return word_model.cuda()


if __name__ == '__main__':
    config = get_args()
    word_model_name = config.word_model
    data_val, data_test = load_val_test_data()
    dictionary = pd.read_pickle(p.dict_path)
    word2idx = dictionary.word2idx
    val_grouped = data_val.groupby("id")
    test_grouped = data_test.groupby("id")

    word_model = load_model(config, dictionary.word2idx)
    print(word_model)
    for combo in ['majority', 'avg']:
        evaluate(combo)