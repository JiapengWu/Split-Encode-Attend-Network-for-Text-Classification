import data_loader as dl
from hierarchical_models import *
import time
import math
import parameters as p
import os
from utility import *
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from word_seq_models import *


def train_data(mini_batch, targets):
    word_model.train()
    sent_model.train()
    word_optimizer.zero_grad()
    sent_optimizer.zero_grad()
    y_pred = model_forward(config.para_pooling, word_model, sent_model, mini_batch)

    loss = criterion(y_pred.cuda(), targets)
    loss.backward()

    nn.utils.clip_grad_norm_(word_model.parameters(), config.clip)
    nn.utils.clip_grad_norm_(sent_model.parameters(), config.clip)

    word_optimizer.step()
    sent_optimizer.step()

    return loss.item()


def check_loss(X, y):
    val_loss = []
    for token, label in iterate_mini_batches(X, y, mini_batch_size):
        if torch.cuda.is_available():
            val_loss.append(
                test_data(pad_3d_batch(token).cuda(), Variable(torch.from_numpy(label), requires_grad=False).cuda()))
        else:
            val_loss.append(
                test_data(pad_3d_batch(token),
                          Variable(torch.from_numpy(label), requires_grad=False)))
    # val_loss = list(map(lambda x: x.item(), val_loss))
    return np.mean(np.array(val_loss))


def test_accuracy_full_batch(X, y):
    preds = []
    labels = []
    val_loss = []
    g = gen_3d_mini_batch(X, y, mini_batch_size)
    for token, label in g:
        soft_pred = model_forward(config.para_pooling, word_model, sent_model, token)
        _, y_pred = torch.max(soft_pred, 1)
        loss = criterion(soft_pred, label)
        val_loss.append(loss.item())
        preds.append(np.ndarray.flatten(y_pred.data.cpu().numpy()))
        labels.append(np.ndarray.flatten(label.data.cpu().numpy()))
    preds = np.array([item for sublist in preds for item in sublist])
    labels = np.array([item for sublist in labels for item in sublist])
    num_correct = sum(preds == labels)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    return np.mean(np.array(val_loss)), float(num_correct)/len(preds), precision, recall, f1, confusion_matrix(labels, preds)


def test_accuracy_mini_batch(tokens, labels):
    y_pred = model_forward(config.para_pooling, word_model, sent_model, tokens)
    _, y_pred = torch.max(y_pred, 1)
    correct = np.ndarray.flatten(y_pred.data.cpu().numpy())
    labels = np.ndarray.flatten(labels.data.cpu().numpy())
    num_correct = sum(correct == labels)
    return float(num_correct) / len(correct)


def test_data(mini_batch, targets):
    y_pred = model_forward(config.para_pooling, word_model, sent_model, mini_batch)
    loss = criterion(y_pred.cuda(), targets)
    return loss.item()


def train_early_stopping():
    best_val_loss = None
    best_acc = None
    loss_epoch = []
    epoch_counter = initial_epoch
    # print(X_train)
    g = gen_3d_mini_batch(X_train, y_train, mini_batch_size)
    batch_start = time.time()
    for i in range(1, num_batch + 1):
        try:
            tokens, labels = next(g)
            # print(tokens.shape)
            loss = train_data(tokens, labels)
            # accuracy_epoch.append(acc)
            loss_epoch.append(loss)
            # print loss every n passes
            number_batchs = len(X_train) // mini_batch_size + 1
            if i % print_loss_every == 0:
                print('| epoch   %d | %d/%d batches | ms/batch (%s) | loss %f' % (
                    epoch_counter, i % (number_batchs + 1), number_batchs, time_since(batch_start), np.mean(loss_epoch)))
                batch_start = time.time()
        except StopIteration:
            word_model.eval()
            sent_model.eval()
            print('-' * 89)
            val_loss, val_acc, precision, recall, f1, conf_matrix = test_accuracy_full_batch(X_val, y_val)
            print('| val set result | valid loss (pure) {:5.4f} | Acc {:8.4f} | Precision {:8.4f} | Recall {:8.4f} '
                  '| F1-score {:8.4f}'.format(val_loss, val_acc, precision, recall, f1))
            print('The confusion matrix is: ')
            print(str(conf_matrix))
            print('-' * 89)

            test_loss, test_acc, precision, recall, f1, conf_matrix = test_accuracy_full_batch(X_test, y_test)
            print('| test set result | valid loss (pure) {:5.4f} | Acc {:8.4f} | Precision {:8.4f} | Recall {:8.4f} '
                  '| F1-score {:8.4f}'.format(test_loss, test_acc, precision, recall, f1))
            print('The confusion matrix is: ')
            print(str(conf_matrix))
            print('-' * 89)

            directory = "./experiments/%s/models/" % config.exp_num

            if not os.path.exists(directory):
                os.makedirs(directory)

            if not best_val_loss or val_loss < best_val_loss:
                best_val_loss = val_loss
            else:  # if loss doesn't go down, divide the learning rate by 5.
                for param_group in sent_optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5
                for param_group in word_optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5
            if not best_acc or val_acc > best_acc:
                with open(directory + 'word_attn.best_acc.pt', 'wb') as f:
                    torch.save(word_model.state_dict(), f)
                with open(directory + 'sent_attn.best_acc.pt', 'wb') as f:
                    torch.save(sent_model.state_dict(), f)
                best_acc = val_acc

            with open(directory + 'word_attn.epoch-{:02d}.pt'.format(epoch_counter), 'wb') as f:
                torch.save(word_model.state_dict(), f)
            with open(directory + 'sent_attn.epoch-{:02d}.pt'.format(epoch_counter), 'wb') as f:
                torch.save(sent_model.state_dict(), f)

            # save for resume
            with open("./experiments/{}/word_optimizer.pt".format(config.exp_num), 'wb') as f:
                torch.save(word_optimizer.state_dict(), f)
            with open("./experiments/{}/sent_optimizer.pt".format(config.exp_num), 'wb') as f:
                torch.save(sent_optimizer.state_dict(), f)

            epoch_counter += 1
            g = gen_3d_mini_batch(X_train, y_train, mini_batch_size)
            loss_epoch = []


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


if __name__ == '__main__':
    if torch.cuda.is_available():
        print("GPU available!")
    else:
        print("Using CPU...")
    dataset, word2idx, idx2word = dl.load_data_set()
    dataset.text2idx(word2idx)
    X_train, y_train, X_val, y_val, X_test, y_test = \
        dataset.X_train, dataset.y_train, dataset.X_val, dataset.y_val, dataset.X_test, dataset.y_test
    # indexing
    print("Text to index finished.")

    config = get_args()
    config.levels = 2
    config.class_number = len(np.unique(y_train))
    print(config)
    config = config
    mini_batch_size = p.batch_size
    initial_epoch = 0

    learning_rate = config.lr
    momentum = config.momentum

    if p.weighted_class:
        # TODO: calculate class weight
        criterion = nn.NLLLoss(torch.FloatTensor(p.class_weight).cuda())
    else:
        criterion = nn.NLLLoss()

    print_loss_every = p.print_loss_every

    if config.word_pooling == 'attn': word_model = AttentionWordRNN(word2idx, config)
    else: word_model = WordEncoder(word2idx, config)

    if config.sent_pooling == 'attn': sent_model = AttentionSentRNN(config)
    else: sent_model = SentEncoder(config)

    if config.resume:
        from functools import reduce
        import re
        from os import listdir
        from os.path import isfile, join

        mypath = "./experiments/{}/models/".format(config.exp_num)
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        try:
            cur_epoch = 1 + int(max(list(reduce(lambda x, y: x + y, [re.findall(r'\d+', x) for x in onlyfiles]))))

            print("Continuing training...")
            pre_word_attn = torch.load(
                "./experiments/{}/models/word_attn.epoch-{:02d}.pt".format(config.exp_num, cur_epoch - 1))
            pre_sent_attn = torch.load(
                "./experiments/{}/models/sent_attn.epoch-{:02d}.pt".format(config.exp_num, cur_epoch - 1))
            word_model.load_state_dict(pre_word_attn)
            sent_model.load_state_dict(pre_sent_attn)
        except ValueError:
            cur_epoch = 0
        initial_epoch = cur_epoch

    word_model.cuda()
    sent_model.cuda()

    print(word_model)
    print(sent_model)

    if config.optimizer.lower() == "adam":
        word_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, word_model.parameters()), lr=config.lr)
        sent_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, sent_model.parameters()), lr=config.lr)
    elif config.optimizer.lower() == "sgd":
        word_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, word_model.parameters()),
                                         lr=config.lr, momentum=config.momentum)
        sent_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, sent_model.parameters()),
                                         lr=config.lr, momentum=config.momentum)
    else:
        raise Exception("Please choose a input optimizer.")

    if config.resume:
        pre_word_optim = torch.load("./experiments/{}/word_optimizer.pt".format(config.exp_num))
        pre_sent_optim = torch.load("./experiments/{}/sent_optimizer.pt".format(config.exp_num))
        word_optimizer.load_state_dict(pre_word_optim)
        sent_optimizer.load_state_dict(pre_sent_optim)

    num_batch = (len(X_train) // mini_batch_size + 2) * (p.num_epoch - initial_epoch)

    train_early_stopping()