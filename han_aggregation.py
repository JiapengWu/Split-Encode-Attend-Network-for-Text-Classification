import utility
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import parameters as p
import pandas as pd
from dataset import TextDataset
from utility import *
import matplotlib
matplotlib.use('agg')
from hierarchical_models import *
import os


def load_val_test_data():
    data_val = pd.read_json(p.sent_split_dir + "val_paragraph_sentence.json")
    data_test = pd.read_json(p.sent_split_dir + "test_paragraph_sentence.json")
    if config.dataset == "yelp":
        data_val['label'] = data_val['label'].apply(lambda x: 0 if x == 1 else 1).astype('int64')
        data_test['label'] = data_test['label'].apply(lambda x: 0 if x == 1 else 1).astype('int64')
    return data_val, data_test


def test_accuracy_full_batch(tokens, labels, word_attn, sent_attn):
    preds = None
    soft_preds = None
    tokens, labels = process_batch(tokens, labels)
    y_soft_pred = model_forward(config.pooling, word_attn, sent_attn, tokens)
    _, y_pred = torch.max(y_soft_pred, 1)
    return preds, soft_preds


def get_prediction(grouped, word2idx, word_attn, sent_attn, combine):
    total_pred = list()
    total_targets = list()
    #
    # id_prob = dict()
    # id_pred = dict()
    # id_label = dict()
    word_attn.eval()
    sent_attn.eval()

    for name, group in grouped:
        tokens = TextDataset._text2idx(group.tokens, word2idx)
        labels = np.array(group.label.values)
        tokens, labels = process_batch(tokens, labels)
        soft_preds = model_forward(config.para_pooling, word_attn, sent_attn, tokens)
        _, prediction = torch.max(soft_preds, 1)

        if combine == 'majority':
            document_pred = 1 if torch.nonzero(prediction).size(0) >= prediction.shape[0] / 2 else 0
        elif combine == 'avg':
            document_pred = 1 if torch.mean(torch.exp(soft_preds), 0)[0].item() < 0.5 else 0
        target = labels[0].item()

        # id_prob[name] = torch.transpose(torch.exp(soft_preds), 0, 1)[0].tolist()
        # id_pred[name] = document_pred
        # id_label[name] = target

        total_pred.append(document_pred)
        total_targets.append(target)

    return total_pred, total_targets


def evaluate(combine='majority'):
    # data_train = pd.read_json(p.sent_split_dir + "val_clustered_sent_split.json")
    data_val, data_test = load_val_test_data()
    dictionary = pd.read_pickle(p.dict_path)
    val_grouped = data_val.groupby("id")
    test_grouped = data_test.groupby("id")

    word_encoder, sent_encoder = load_model(config, dictionary.word2idx)
    print(word_encoder)
    print(sent_encoder)

    """evaluate the model """
    total_pred, total_targets = \
        get_prediction(val_grouped, dictionary.word2idx, word_encoder, sent_encoder, combine)
    total_pred = np.array(total_pred)
    total_targets = np.array(total_targets)
    precision, recall, f1, _ = precision_recall_fscore_support(total_targets, total_pred)
    acc = accuracy_score(total_targets, total_pred)
    conf_matrix = confusion_matrix(total_targets, total_pred)
    print(combine + " | val set Precision: " + str(precision) + " | Recall: " + str(recall) + " | F1-score: " + str(f1) +
          " | Accuracy: " + str(acc))
    print("Confusion matrix: \n" + str(conf_matrix))


    total_pred, total_targets = \
        get_prediction(test_grouped, dictionary.word2idx, word_encoder, sent_encoder, combine)
    total_pred = np.array(total_pred)
    total_targets = np.array(total_targets)
    precision, recall, f1, _ = precision_recall_fscore_support(total_targets, total_pred)
    acc = accuracy_score(total_targets, total_pred)
    conf_matrix = confusion_matrix(total_targets, total_pred)
    print(combine + " |test set Precision: " + str(precision) + " | Recall: " + str(recall) + " | F1-score: " + str(f1) +
          " | Accuracy: " + str(acc))
    print("Confusion matrix: \n" + str(conf_matrix))


def process_batch(tokens, labels):
    tokens = pad_3d_batch(tokens)
    return tokens.cuda(), Variable(torch.from_numpy(labels), requires_grad=False).cuda()


def load_model(config, word2idx):
    word_encoder = AttentionWordRNN(word2idx, config)
    sent_encoder = AttentionSentRNN(config)

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
    pre_word_encoder = torch.load(
        "./experiments/{}/models/word_attn.epoch-{:02d}.pt".format(config.model_exp_num, index))
    pre_sent_encoder = torch.load(
        "./experiments/{}/models/sent_attn.epoch-{:02d}.pt".format(config.model_exp_num, index))

    word_encoder.load_state_dict(pre_word_encoder)
    sent_encoder.load_state_dict(pre_sent_encoder)
    return word_encoder.cuda(), sent_encoder.cuda()


if __name__ == '__main__':
    config = get_args()
    # for combo in ['majority', 'avg']:
    evaluate('majority')