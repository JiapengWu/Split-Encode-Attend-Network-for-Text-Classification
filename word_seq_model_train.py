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
    model.train()
    optimizer.zero_grad()
    logits, hidden = model.forward(mini_batch)
    loss = criterion(logits.cuda(), targets)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), config.clip)
    optimizer.step()
    return loss.item()


def test_accuracy_full_batch(X, y):
    preds = []
    labels = []
    val_loss = []
    g = gen_2d_mini_batch(X, y, mini_batch_size)
    for token, label in g:
        logits, hidden = model.forward(token)
        y_pred = torch.max(logits, 1)[1]
        loss = criterion(logits.cuda(), label)
        val_loss.append(loss.item())
        preds.append(np.ndarray.flatten(y_pred.data.cpu().numpy()))
        labels.append(np.ndarray.flatten(label.data.cpu().numpy()))
    preds = np.array([item for sublist in preds for item in sublist])
    labels = np.array([item for sublist in labels for item in sublist])
    num_correct = sum(preds == labels)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    return np.mean(np.array(val_loss)), float(num_correct)/len(preds), precision, recall, f1, confusion_matrix(labels, preds)


def train_early_stopping():
    best_val_loss = None
    best_acc = None
    loss_epoch = []
    epoch_counter = initial_epoch
    g = gen_2d_mini_batch(X_train, y_train, mini_batch_size)
    batch_start = time.time()
    for i in range(1, num_batch + 1):
        try:
            tokens, labels = next(g)
            loss = train_data(tokens, labels)
            loss_epoch.append(loss)
            # print loss every n passes
            number_batchs = len(X_train) // mini_batch_size + 1
            if i % print_loss_every == 0:
                print('| epoch   %d | %d/%d batches | ms/batch (%s) | loss %f' % (
                    epoch_counter, i % (number_batchs + 1), number_batchs, time_since(batch_start), np.mean(loss_epoch)))
                batch_start = time.time()
        except StopIteration:
            model.eval()
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
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.2
            if not best_acc or val_acc > best_acc:
                with open(directory + 'word_model.best_acc.pt', 'wb') as f:
                    torch.save(model.state_dict(), f)
                best_acc = val_acc

            with open(directory + 'word_model.epoch-{:02d}.pt'.format(epoch_counter), 'wb') as f:
                torch.save(model.state_dict(), f)

            # save for resume
            with open("./experiments/{}/optimizer.pt".format(config.exp_num), 'wb') as f:
                torch.save(optimizer.state_dict(), f)

            epoch_counter += 1
            g = gen_2d_mini_batch(X_train, y_train, mini_batch_size)
            loss_epoch = []
            batch_start = time.time()


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
        criterion = nn.CrossEntropyLoss(torch.FloatTensor(p.class_weight).cuda())
    else:
        criterion = nn.CrossEntropyLoss()

    print_loss_every = p.print_loss_every
    if config.encoder_type.lower() == "cnn":
        if config.use_pyramid: model = DPCNN(config, word2idx)
        else: model = CNN_Text(config, word2idx)
    else:
        model = RNNPooling(config, word2idx)

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
                "./experiments/{}/models/word_model.epoch-{:02d}.pt".format(config.exp_num, cur_epoch - 1))
            model.load_state_dict(pre_word_attn)
        except ValueError:
            cur_epoch = 0

        initial_epoch = cur_epoch

    model.cuda()
    print(model)

    if config.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)
    elif config.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr, momentum=config.momentum)
    else:
        raise Exception("Please choose a optimizer: adam or sgd")

    if config.resume:
        pre_optim = torch.load("./experiments/{}/optimizer.pt".format(config.exp_num))
        optimizer.load_state_dict(pre_optim)

    num_batch = (len(X_train) // mini_batch_size + 2) * (p.num_epoch - initial_epoch)

    train_early_stopping()