from data_loader import assign_embedding
from torch.nn.functional import softmax
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class RNNPooling(nn.Module):
    def __init__(self, config, word2idx):
        super(RNNPooling, self).__init__()
        self.num_token = len(word2idx)
        self.word_pooling = config.word_pooling
        self.embed_size = config.emsize
        self.n_classes = config.class_number
        self.word_gru_hidden = config.word_gru_hidden
        self.lookup = nn.Embedding(self.num_token, self.embed_size)
        self.dropout = nn.Dropout(config.dropout)
        self.word2idx = word2idx
        self.encoder_type = config.encoder_type
        self.final_linear = nn.Linear(2 * self.word_gru_hidden, self.n_classes)
        if config.use_glove:
            weights_matrix = assign_embedding(word2idx, self.embed_size)
            weights_matrix = torch.from_numpy(weights_matrix).float()
            assert self.num_token == weights_matrix.shape[0], self.embed_size == weights_matrix.shape[1]
            self.lookup.weight = nn.Parameter(weights_matrix)

        if config.encoder_type == 'LSTM':
            self.word_encoder = nn.LSTM(self.embed_size, self.word_gru_hidden, bidirectional=True)
        elif config.encoder_type == 'GRU':
            self.word_encoder = nn.GRU(self.embed_size, self.word_gru_hidden, bidirectional=True)

    def forward(self, embed):
        state_word = self.init_hidden(embed.shape[1])
        embedded = self.lookup(embed)  # [wlen, bsz, esz]
        output_word, state_word = self.word_encoder(self.dropout(embedded), state_word)  # [wlen, bsz, nhid*2]
        if self.word_pooling == 'mean':
            word_vectors = torch.mean(output_word, 0)  # [bsz, 2*hid]
        elif self.word_pooling == 'max':
            word_vectors = torch.max(output_word, 0)[0]  # [bsz, 2*hid]
        logit = self.final_linear(word_vectors)
        return logit, word_vectors

    def init_hidden(self, batch_size):
        if self.encoder_type == 'GRU':
            return Variable(torch.zeros(2, batch_size, self.word_gru_hidden)).cuda()
        elif self.encoder_type == 'LSTM':
            return (Variable(torch.zeros(2, batch_size, self.word_gru_hidden)).cuda(),
                    Variable(torch.zeros(2, batch_size, self.word_gru_hidden)).cuda())


class CNN_Text(nn.Module):
    def __init__(self, config, word2idx):
        super(CNN_Text, self).__init__()
        self.config = config
        self.num_tokens = len(word2idx)
        self.embed_size = config.emsize
        self.n_classes = config.class_number
        self.kernel_num = config.kernel_num
        self.kernel_sizes = list(map(lambda x: int(x), config.kernel_sizes.split(",")))
        self.embedding = nn.Embedding(self.num_tokens, self.embed_size)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, self.kernel_num, [window_size, self.embed_size], padding=(window_size - 1, 0))
            for window_size in self.kernel_sizes
        ])

        if config.use_glove:
            weights_matrix = assign_embedding(word2idx, self.embed_size)
            weights_matrix = torch.from_numpy(weights_matrix).float()
            assert self.num_tokens == weights_matrix.shape[0], self.embed_size == weights_matrix.shape[1]
            self.embedding.weight = nn.Parameter(weights_matrix, requires_grad=True)

        self.dropout = nn.Dropout(self.config.dropout)
        self.fc = nn.Linear(len(self.kernel_sizes) * self.kernel_num, self.n_classes)

    def forward(self, x):
        debug = False
        x = x.transpose(0, 1)
        x = self.embedding(x)
        x = x.view(x.size(0), 1, x.size(1), x.size(2))  # (batch_size, 1, length, embed_size)
        # Apply a convolution + max pool layer for each window size
        xs = []
        for conv in self.convs:
            # try:
            x2 = F.relu(conv(x))        # [B, F, T, 1]
            # except:
            #     print(x)
            #     print(x.shape)
            x2 = torch.squeeze(x2, -1)  # [B, F, T]
            x2 = F.max_pool1d(x2, x2.size(2))  # [B, F, 1]
            xs.append(x2)
        x = torch.cat(xs, 2)            # [B, F, window]
        # FC
        x = x.view(x.size(0), -1)       # [B, F * window]
        logits = self.fc(self.dropout(x))             # [B, class]

        return logits, x


class DPCNN(nn.Module):
    """
    DPCNN for sentences classification.
    """
    def __init__(self, config, word2idx):
        super(DPCNN, self).__init__()
        self.config = config
        self.num_tokens = len(word2idx)
        self.embed_size = config.emsize
        self.embed = nn.Embedding(self.num_tokens, self.embed_size)
        self.n_classes = config.class_number
        self.kernel_num = config.kernel_num
        self.conv_region_embedding = nn.Conv2d(1, self.kernel_num, (3, self.embed_size), stride=1)
        self.conv3 = nn.Conv1d(self.kernel_num, self.kernel_num, kernel_size=3, stride=1)
        self.pooling = nn.MaxPool1d(kernel_size=3, stride=2)
        # self.padding_conv = nn.ZeroPad2d((0, 0, 1, 1))
        # self.padding_pool = nn.ZeroPad2d((0, 0, 0, 1))
        self.padding_conv = nn.ZeroPad2d((1, 1, 0, 0))
        self.padding_pool = nn.ZeroPad2d((1, 0, 0, 0))
        self.act_fun = nn.Tanh()
        self.linear_out = nn.Linear(self.kernel_num, self.n_classes)
        self.dropout = nn.Dropout(self.config.dropout)

        if config.use_glove:
            weights_matrix = assign_embedding(word2idx, self.embed_size)
            weights_matrix = torch.from_numpy(weights_matrix).float()
            assert self.num_tokens == weights_matrix.shape[0], self.embed_size == weights_matrix.shape[1]
            self.embed.weight = nn.Parameter(weights_matrix, requires_grad=True)

    def forward(self, x):
        debug = False
        x = x.transpose(0, 1)
        # print(x.shape)
        x = self.embed(x) # (batch_size, length, embed_size)
        # print(x.shape)
        x = x.view(x.size(0), 1, x.size(1), x.size(2))  # (batch_size, 1, length, embed_size)
        # print(x.shape)
        batch = x.shape[0]

        # Region embedding
        x = self.conv_region_embedding(x).squeeze(-1)        # [batch_size, channel_size, length]
        x = self.padding_conv(x)    # [batch_size, channel_size, length + 2]
        x = F.relu(x)
        x = self.conv3(x)   # [batch_size, channel_size, length]
        x = self.padding_conv(x)
        x = F.relu(x)
        x = self.conv3(x)

        while x.size()[-1] >= 2:
            x = self._block(x)  # [batch_size, channel_size, length / 2]
        x = x.view(batch, self.kernel_num)
        logit = self.linear_out(self.dropout(x))
        if debug: print(F.softmax(logit, dim=1))

        return logit, x

    def _block(self, x):
        # Pooling
        x = self.padding_pool(x)
        px = F.max_pool1d(x, kernel_size=3, stride=2) # [batch_size, channel_size, length / 2]
        # Convolution
        x = self.padding_conv(px)
        x = F.relu(x)
        x = self.conv3(x)

        x = self.padding_conv(x)
        x = F.relu(x)
        x = self.conv3(x)

        # Short Cut
        x = x + px

        return x

    def predict(self, x):
        self.eval()
        out = self.forward(x)
        predict_labels = torch.max(out, 1)[1]
        self.train(mode=True)
        return predict_labels


class CharCNN(nn.Module):
    def __init__(self, args):
        super(CharCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(args.num_features, 256, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=7, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8704, 1024),
            nn.ReLU(),
            nn.Dropout(p=args.dropout)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=args.dropout)
        )

        self.fc3 = nn.Linear(1024, 2)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        # collapse
        x = x.view(x.size(0), -1)
        # linear layer
        x = self.fc1(x)
        # linear layer
        x = self.fc2(x)
        # linear layer
        x = self.fc3(x)
        # output layer
        x = self.log_softmax(x)

        return x