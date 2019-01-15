import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from data_loader import assign_embedding
from torch.nn.functional import softmax


def batch_matmul_bias(seq, weight, bias, nonlinearity='tanh'):
    s = None
    bias_dim = bias.size()
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        _s_bias = _s + bias.expand(bias_dim[0], _s.size()[0]).transpose(0,1)
        if(nonlinearity=='tanh'):
            _s_bias = torch.tanh(_s_bias)
        _s_bias = _s_bias.unsqueeze(0)
        if(s is None):
            s = _s_bias
        else:
            s = torch.cat((s,_s_bias), 0)
    return s


def batch_matmul(seq, weight, nonlinearity='tanh'):
    s = None
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        if(nonlinearity=='tanh'):
            _s = torch.tanh(_s)
        _s = _s.unsqueeze(0)
        if(s is None):
            s = _s
        else:
            s = torch.cat((s,_s),0)
    return s.squeeze(2)


def attention_mul(rnn_outputs, att_weights):
    attn_vectors = None
    for i in range(rnn_outputs.size(0)):
        h_i = rnn_outputs[i]
        a_i = att_weights[i].unsqueeze(1).expand_as(h_i)
        h_i = a_i * h_i
        h_i = h_i.unsqueeze(0)
        if(attn_vectors is None):
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors,h_i),0)
    return torch.sum(attn_vectors, 0)


# ## Word attention model with bias
class AttentionWordRNN(nn.Module):
    
    def __init__(self, word2idx, config):

        super(AttentionWordRNN, self).__init__()

        self.embed_size = config.emsize
        self.config = config
        self.word_gru_hidden = config.word_gru_hidden
        self.num_tokens = len(word2idx)
        self.lookup = nn.Embedding(self.num_tokens, self.embed_size)
        self.dropout = nn.Dropout(config.dropout)
        self.word2idx = word2idx
        self.encoder_type = config.encoder_type

        if config.use_glove:
            weights_matrix = assign_embedding(word2idx, self.embed_size)
            weights_matrix = torch.from_numpy(weights_matrix).float()
            assert self.num_tokens == weights_matrix.shape[0], self.embed_size == weights_matrix.shape[1]
            self.lookup.weight = nn.Parameter(weights_matrix)

        if config.encoder_type == 'LSTM':
            self.word_encoder = nn.LSTM(self.embed_size, self.word_gru_hidden, bidirectional=True)
        elif config.encoder_type == 'GRU':
            self.word_encoder = nn.GRU(self.embed_size, self.word_gru_hidden, bidirectional=True)

        self.weight_W_word = nn.Parameter(torch.Tensor(2 * self.word_gru_hidden, 2 * self.word_gru_hidden))
        self.bias_word = nn.Parameter(torch.Tensor(2 * self.word_gru_hidden, 1))
        self.weight_proj_word = nn.Parameter(torch.Tensor(2 * self.word_gru_hidden, 1))

        self.softmax_word = nn.Softmax(dim=1)
        self.weight_W_word.data.uniform_(-0.1, 0.1)
        self.weight_proj_word.data.uniform_(-0.1,0.1)
        
    def forward(self, embed, state_word):
        # embed: # [wlen, bsz]
        embedded = self.lookup(embed)  # [wlen, bsz, esz]
        # self.word_gru.flatten_parameters()
        output_word, state_word = self.word_encoder(self.dropout(embedded), state_word)  # [wlen, bsz, nhid*2]
        word_squish = batch_matmul_bias(output_word, self.weight_W_word, self.bias_word, nonlinearity='tanh')  # [wlen, bsz, nhid*2]
        word_attn = batch_matmul(word_squish, self.weight_proj_word)  # [wlen, bsz]
        # print(embed == self.word2idx['<pad>'])
        word_attn_norm = self.softmax_word(word_attn.transpose(1, 0))  # [bsz, wlen]
        output_word = self.filter_outp(embed, output_word)
        word_attn_vectors = attention_mul(output_word, word_attn_norm.transpose(1, 0))  # [bsz, nhid*2]
        # print(embed.shape, embedded.shape, output_word.shape, word_squish.shape, word_attn.shape, word_attn_norm.shape, word_attn_vectors.shape)
        return word_attn_vectors, state_word, word_attn_norm

    def filter_outp(self, embed, output_word):
        embed_expand = embed.unsqueeze(2).repeat(1, 1, output_word.shape[2]).cuda()
        zeros = torch.zeros(output_word.shape).cuda()
        return torch.where(embed_expand == self.word2idx['<pad>'], zeros, output_word)

    def init_hidden(self, batch_size):
        if self.encoder_type == 'GRU':
            return Variable(torch.zeros(2, batch_size, self.word_gru_hidden)).cuda()
        elif self.encoder_type == 'LSTM':
            return (Variable(torch.zeros(2, batch_size, self.word_gru_hidden)).cuda(),
                    Variable(torch.zeros(2, batch_size, self.word_gru_hidden)).cuda())


class AttentionSentRNN(nn.Module):
    
    def __init__(self, config):
        
        super(AttentionSentRNN, self).__init__()

        self.n_classes = config.class_number
        self.word_gru_hidden = config.word_gru_hidden
        self.sent_gru_hidden = config.sent_gru_hidden
        self.dropout = nn.Dropout(config.dropout)
        self.encoder_type = config.encoder_type

        if config.encoder_type == 'LSTM':
            self.sent_encoder = nn.LSTM(2 * self.word_gru_hidden, self.sent_gru_hidden, bidirectional=True)
        elif config.encoder_type == 'GRU':
            self.sent_encoder = nn.GRU(2 * self.word_gru_hidden, self.sent_gru_hidden, bidirectional=True)

        self.weight_W_sent = nn.Parameter(torch.Tensor(2 * self.sent_gru_hidden , 2 * self.sent_gru_hidden))
        self.bias_sent = nn.Parameter(torch.Tensor(2 * self.sent_gru_hidden, 1))
        self.weight_proj_sent = nn.Parameter(torch.Tensor(2 * self.sent_gru_hidden, 1))
        self.final_linear = nn.Linear(2 * self.sent_gru_hidden, self.n_classes)

        self.softmax_sent = nn.Softmax(dim=1)
        self.weight_W_sent.data.uniform_(-0.1, 0.1)
        self.weight_proj_sent.data.uniform_(-0.1,0.1)

    def forward(self, batch_size, word_attention_vectors, state_sent):
        max_sent_count = int(word_attention_vectors.shape[0] / batch_size)
        word_attention_vectors = word_attention_vectors.view(max_sent_count, batch_size, -1)  # [slen, bsz, nhid*2]
        # self.sent_gru.flatten_parameters()
        output_sent, state_sent = self.sent_encoder(self.dropout(word_attention_vectors), state_sent)  #[slen, bsz, nhid*2]
        sent_squish = batch_matmul_bias(output_sent, self.weight_W_sent,self.bias_sent,
                                        nonlinearity='tanh')  # [slen, bsz, nhid*2]
        sent_attn = batch_matmul(sent_squish, self.weight_proj_sent)  # [slen, bsz]
        sent_attn = self.filter_sent_attn(word_attention_vectors, sent_attn)
        sent_attn_norm = self.softmax_sent(sent_attn.transpose(1,0))  # [bsz, slen]
        sent_attn_vectors = attention_mul(output_sent, sent_attn_norm.transpose(1,0))  # [bsz, nhid*2]
        # final classifier
        # final_map = self.final_linear(sent_attn_vectors.squeeze(0)) #[bsz, n_class]
        final_map = self.final_linear(sent_attn_vectors) #[bsz, n_class]
        # print(word_attention_vectors.shape, output_sent.shape, sent_squish.shape, sent_attn.shape, sent_attn_norm.shape, \
        #       sent_attn_vectors.shape, final_map.shape)
        return F.log_softmax(final_map, dim=1), state_sent, sent_attn_vectors

    def filter_sent_attn(self, word_attention_vectors, sent_attn):
        word_attention_max = torch.max(word_attention_vectors, dim=2)[0]
        mask = torch.tensor(-10000000).repeat(sent_attn.shape).float().cuda()
        return torch.where(word_attention_max == 0, mask, sent_attn)

    def init_hidden(self, batch_size):
        if self.encoder_type == 'GRU':
            return Variable(torch.zeros(2, batch_size, self.sent_gru_hidden)).cuda()
        elif self.encoder_type == 'LSTM':
            return (Variable(torch.zeros(2, batch_size, self.sent_gru_hidden)).cuda(),
                    Variable(torch.zeros(2, batch_size, self.sent_gru_hidden)).cuda())


class ParagraphAttention(nn.Module):

    def __init__(self, config, word_attn, sent_attn):
        super(ParagraphAttention, self).__init__()
        self.config = config
        self.word_attn = word_attn
        self.sent_attn = sent_attn
        self.para_gru_hidden = config.para_gru_hidden
        self.para_pooling = config.para_pooling
        self.attention_hidden = sent_attn.sent_gru_hidden
        self.n_classes = config.class_number
        self.dropout = nn.Dropout(config.dropout)
        self.encoder_type = config.encoder_type

        if config.encoder_type == 'LSTM':
            self.para_encoder = nn.LSTM(2 * self.attention_hidden, self.para_gru_hidden, bidirectional=True)
        elif config.encoder_type == 'GRU':
            self.para_encoder = nn.GRU(2 * self.attention_hidden, self.para_gru_hidden, bidirectional=True)

        self.ws1 = nn.Linear(2 * self.para_gru_hidden, 2 * self.attention_hidden, bias=True)
        self.ws2 = nn.Linear(2 * self.attention_hidden, 1, bias=False)
        self.final_linear = nn.Linear(2 * self.para_gru_hidden, self.n_classes)
        self.tanh = nn.Tanh()
        self.ws1.weight.data.uniform_(-0.1, 0.1)
        self.ws2.weight.data.uniform_(-0.1, 0.1)

    def forward(self, mini_batch):
        max_sents, batch_size, max_tokens = mini_batch.size()
        state_word = self.word_attn.init_hidden(batch_size)
        state_sent = self.sent_attn.init_hidden(batch_size)
        state_para = self.init_hidden()
        # disable dropout if necessary
        s = None
        for i in range(max_sents):
            _s, state_word, _ = self.word_attn(mini_batch[i, :, :].transpose(0, 1), state_word)
            if (s is None):
                s = _s
            else:
                s = torch.cat((s, _s), 0)

        y_pred, state_sent, sent_attn_vectors = self.sent_attn(batch_size, s, state_sent)  # sent_attn_vectors: [plen, nhid*2]
        # self.para_gru.flatten_parameters()

        output_para, state_para = self.para_encoder(self.dropout(sent_attn_vectors.unsqueeze(1)), state_para)  # [plen, 1, nhid*2]
        # para_squish = self.tanh(self.ws1(self.dropout(output_para)))  # [plen, 1, nhid*2]
        # para_attn = self.ws2(para_squish)  # [plen, 1, 1]
        # para_attn_norm = softmax(para_attn, dim=0)  # [plen, 1, 1]
        # # print(output_para.shape, para_attn_norm.shape)
        # para_attn_vectors = torch.mm(para_attn_norm.squeeze(2).transpose(0, 1), output_para.squeeze(1)) # [1, nhid*2]
        # final_map = self.final_linear(para_attn_vectors)  # [1, n_class]
        # return F.log_softmax(final_map, dim=1), state_para, para_attn_norm

        if self.para_pooling == 'attn':
            if mini_batch.shape[1] == 1:
                para_attn_norm = torch.tensor([[[1.0]]])
                para_vectors = output_para.squeeze(1)
            else:
                para_squish = self.tanh(self.ws1(self.dropout(output_para)))  # [plen, 1, nhid*2]
                para_attn = self.ws2(para_squish)  # [plen, 1, 1]
                para_attn_norm = softmax(para_attn, dim=0)  # [plen, 1, 1]
                # print(output_para.shape, para_attn_norm.shape)
                para_vectors = torch.mm(para_attn_norm.squeeze(2).transpose(0, 1), output_para.squeeze(1))  # [1, nhid*2]
            final_map = self.final_linear(para_vectors)  # [1, n_class]
            return F.log_softmax(final_map, dim=1), para_attn_norm

        elif self.para_pooling == 'ensem-attn':
            if mini_batch.shape[1] == 1: return y_pred

            y_pred = torch.exp(y_pred)  # [plen, 1]
            para_squish = self.tanh(self.ws1(self.dropout(output_para.squeeze(1))))  # [plen, nhid*2]
            # print(para_squish.shape)
            para_attn = self.ws2(para_squish)  # [plen, 1]
            para_attn_norm = softmax(para_attn, dim=0)  # [plen, 1]
            doc_pred = torch.mm(para_attn_norm.transpose(0, 1), y_pred)  # [1, plen]
            # print(doc_pred)
            return torch.log(doc_pred)

        elif self.para_pooling == 'mean':
            para_vectors = torch.mean(output_para, 0)
        elif self.para_pooling == 'max':
            para_vectors = torch.max(output_para, 0)[0]  # [1, 2*hid]
        final_map = self.final_linear(para_vectors)  # [1, n_class]
        return F.log_softmax(final_map, dim=1)

    def init_hidden(self):
        if self.encoder_type == 'GRU':
            return Variable(torch.zeros(2, 1, self.para_gru_hidden)).cuda()
        elif self.encoder_type == 'LSTM':
            return (Variable(torch.zeros(2, 1, self.para_gru_hidden)).cuda(),
                    Variable(torch.zeros(2, 1, self.para_gru_hidden)).cuda())


class ParagraphEnsemble(nn.Module):

    def __init__(self, config, word_attn, sent_attn):
        super(ParagraphEnsemble, self).__init__()
        self.config = config
        self.word_attn = word_attn
        self.sent_attn = sent_attn
        self.attention_hidden = sent_attn.sent_gru_hidden
        self.n_classes = config.class_number
        self.dropout = nn.Dropout(config.dropout)
        self.ws1 = nn.Linear(2 * self.attention_hidden, 2 * self.attention_hidden, bias=True)
        self.ws2 = nn.Linear(2 * self.attention_hidden, 1, bias=False)
        self.tanh = nn.Tanh()
        self.ws1.weight.data.uniform_(-0.1, 0.1)
        self.ws2.weight.data.uniform_(-0.1, 0.1)

    def forward(self, mini_batch):
        # disable dropout if necessary
        max_sents, batch_size, max_tokens = mini_batch.size()
        state_word = self.word_attn.init_hidden(batch_size)
        state_sent = self.sent_attn.init_hidden(batch_size)

        s = None
        for i in range(max_sents):
            _s, state_word, _ = self.word_attn(mini_batch[i, :, :].transpose(0, 1), state_word)
            if (s is None):
                s = _s
            else:
                s = torch.cat((s, _s), 0)

        y_pred, state_sent, sent_attn_vectors = self.sent_attn(batch_size, s, state_sent)  # [bsz, nhid*2]
        y_pred = torch.exp(y_pred)
        hbar = self.tanh(self.ws1(self.dropout(sent_attn_vectors)))  # [bsz, nhid*2]
        alphas = softmax(self.ws2(hbar), dim=0)  # [bsz, 1]
        normalised_pred = torch.mm(alphas.transpose(0, 1), y_pred)
        return torch.log(normalised_pred)


class ParagraphEnsembleAttention(nn.Module):
    def __init__(self, config, word_attn, sent_attn):
        super(ParagraphEnsembleAttention, self).__init__()
        self.config = config
        self.word_attn = word_attn
        self.sent_attn = sent_attn
        self.para_gru_hidden = config.para_gru_hidden
        self.attention_hidden = sent_attn.sent_gru_hidden
        self.n_classes = config.class_number
        self.encoder_type = config.encoder_type
        self.dropout = nn.Dropout(config.dropout)

        if config.encoder_type == 'LSTM':
            self.para_encoder = nn.LSTM(2 * self.attention_hidden, self.para_gru_hidden, bidirectional=True)
        elif config.encoder_type == 'GRU':
            self.para_encoder = nn.GRU(2 * self.attention_hidden, self.para_gru_hidden, bidirectional=True)

        self.ws1 = nn.Linear(2 * self.para_gru_hidden, 2 * self.attention_hidden, bias=True)
        self.ws2 = nn.Linear(2 * self.attention_hidden, 1, bias=False)
        self.tanh = nn.Tanh()
        self.ws1.weight.data.uniform_(-0.1, 0.1)
        self.ws2.weight.data.uniform_(-0.1, 0.1)

    def forward(self, mini_batch):
        # disable dropout if necessary
        max_sents, batch_size, max_tokens = mini_batch.size()
        state_word = self.word_attn.init_hidden(batch_size)
        state_sent = self.sent_attn.init_hidden(batch_size)
        state_para = self.init_hidden()

        s = None
        for i in range(max_sents):
            _s, state_word, _ = self.word_attn(mini_batch[i, :, :].transpose(0, 1), state_word)
            if (s is None):
                s = _s
            else:
                s = torch.cat((s, _s), 0)

        y_pred, state_sent, sent_attn_vectors = self.sent_attn(batch_size, s, state_sent)  # [plen, nhid*2]
        y_pred = torch.exp(y_pred) # [plen, 1]

        output_para, state_para = self.para_encoder(self.dropout(sent_attn_vectors.unsqueeze(1)), state_para)  # [plen, 1, nhid*2]

        para_squish = self.tanh(self.ws1(self.dropout(output_para)))  # [plen, 1, nhid*2]
        para_attn = self.ws2(para_squish)  # [plen, 1, 1]
        para_attn_norm = softmax(para_attn, dim=0)  # [plen, 1, 1]
        # print(output_para.shape, para_attn_norm.shape)
        para_attn_vectors = torch.mm(para_attn_norm.squeeze(2).transpose(0, 1), y_pred)  # [1, plen]

        return torch.log(para_attn_vectors)

    def init_hidden(self):
        if self.encoder_type == 'GRU':
            return Variable(torch.zeros(2, 1, self.para_gru_hidden)).cuda()
        elif self.encoder_type == 'LSTM':
            return (Variable(torch.zeros(2, 1, self.para_gru_hidden)).cuda(),
                    Variable(torch.zeros(2, 1, self.para_gru_hidden)).cuda())


class WordEncoder(nn.Module):
    def __init__(self, word2idx, config):

        super(WordEncoder, self).__init__()
        self.num_token = len(word2idx)
        self.word_pooling = config.word_pooling
        self.embed_size = config.emsize
        self.word_gru_hidden = config.word_gru_hidden
        self.lookup = nn.Embedding(self.num_token, self.embed_size)
        self.dropout = nn.Dropout(config.dropout)
        self.word2idx = word2idx
        self.encoder_type = config.encoder_type
        if config.use_glove:
            weights_matrix = assign_embedding(word2idx, self.embed_size)
            weights_matrix = torch.from_numpy(weights_matrix).float()
            assert self.num_token == weights_matrix.shape[0], self.embed_size == weights_matrix.shape[1]
            self.lookup.weight = nn.Parameter(weights_matrix)

        if config.encoder_type == 'LSTM':
            self.word_encoder = nn.LSTM(self.embed_size, self.word_gru_hidden, bidirectional=True)
        elif config.encoder_type == 'GRU':
            self.word_encoder = nn.GRU(self.embed_size, self.word_gru_hidden, bidirectional=True)

    def forward(self, embed, state_word):
        embedded = self.lookup(embed)  # [wlen, bsz, esz]
        output_word, state_word = self.word_encoder(self.dropout(embedded), state_word)  # [wlen, bsz, nhid*2]
        if self.word_pooling == 'mean':
            word_vectors = torch.mean(output_word, 0)  # [bsz, 2*hid]
        elif self.word_pooling == 'max':
            word_vectors = torch.max(output_word, 0)[0]  # [bsz, 2*hid]
        return word_vectors, state_word

    def init_hidden(self, batch_size):
        if self.encoder_type == 'GRU':
            return Variable(torch.zeros(2, batch_size, self.word_gru_hidden)).cuda()
        elif self.encoder_type == 'LSTM':
            return (Variable(torch.zeros(2, batch_size, self.word_gru_hidden)).cuda(),
                    Variable(torch.zeros(2, batch_size, self.word_gru_hidden)).cuda())


class SentEncoder(nn.Module):
    def __init__(self, config):

        super(SentEncoder, self).__init__()
        self.sent_pooling = config.sent_pooling
        self.sent_gru_hidden = config.sent_gru_hidden
        self.word_gru_hidden = config.word_gru_hidden
        self.dropout = nn.Dropout(config.dropout)
        self.levels = config.levels
        self.encoder_type = config.encoder_type
        if config.encoder_type == 'LSTM':
            self.sent_encoder = nn.LSTM(2 * self.word_gru_hidden, self.sent_gru_hidden, bidirectional=True)
        elif config.encoder_type == 'GRU':
            self.sent_encoder = nn.GRU(2 * self.word_gru_hidden, self.sent_gru_hidden, bidirectional=True)
        if config.levels == 2:
            self.n_classes = config.class_number
            self.final_linear = nn.Linear(2 * self.sent_gru_hidden, self.n_classes)

    def forward(self, batch_size, word_vectors, state_sent):
        max_word_count = int(word_vectors.shape[0] / batch_size)

        word_vectors = word_vectors.view(max_word_count, batch_size, -1)
        output_sent, state_sent = self.sent_encoder(self.dropout(word_vectors), state_sent)  # [slen, bsz, nhid*2]
        if self.sent_pooling == 'mean':
            sent_vectors = torch.mean(output_sent, 0)
        elif self.sent_pooling == 'max':
            sent_vectors = torch.max(output_sent, 0)[0]

        if self.levels == 2:
            final_map = self.final_linear(sent_vectors)
            return F.log_softmax(final_map, dim=1)

        return sent_vectors

    def init_hidden(self, batch_size):
        if self.encoder_type == 'GRU':
            return Variable(torch.zeros(2, batch_size, self.sent_gru_hidden)).cuda()
        elif self.encoder_type == 'LSTM':
            return (Variable(torch.zeros(2, batch_size, self.sent_gru_hidden)).cuda(),
                    Variable(torch.zeros(2, batch_size, self.sent_gru_hidden)).cuda())


class ParagraphPooling(nn.Module):
    def __init__(self, config, word_encoder, sent_encoder):
        super(ParagraphPooling, self).__init__()
        self.config = config
        self.word_encoder = word_encoder
        self.sent_encoder = sent_encoder
        self.para_pooling = config.para_pooling
        self.n_classes = config.class_number
        self.sent_gru_hidden = sent_encoder.sent_gru_hidden
        self.dropout = nn.Dropout(config.dropout)
        self.final_linear = nn.Linear(2 * self.para_gru_hidden, self.n_classes)

    def forward(self, mini_batch):
        max_sents, batch_size, max_tokens = mini_batch.size()
        state_word = self.word_encoder.init_hidden(batch_size)
        state_sent = self.sent_encoder.init_hidden(batch_size)

        s = None
        if self.config.word_pooling == 'attn':
            for i in range(max_sents):
                _s, state_word, _ = self.word_encoder(mini_batch[i, :, :].transpose(0, 1), state_word)
                if (s is None):
                    s = _s
                else:
                    s = torch.cat((s, _s), 0)
        else:
            for i in range(max_sents):
                _s, state_word = self.word_encoder(mini_batch[i, :, :].transpose(0, 1), state_word)
                if (s is None):
                    s = _s
                else:
                    s = torch.cat((s, _s), 0)

        if self.config.sent_pooling == 'attn': _, _, sent_vectors = self.sent_encoder(batch_size, s, state_sent)  # sent_vectors: [plen, nhid*2]
        else: sent_vectors = self.sent_encoder(batch_size, s, state_sent)  # sent_vectors: [plen, nhid*2]

        if self.para_pooling == 'mean':
            para_vectors = torch.mean(sent_vectors, 0)
        elif self.para_pooling == 'max':
            para_vectors = torch.max(sent_vectors, 0)[0] # [1, 2*hid]
        final_map = self.final_linear(para_vectors) # [1, n_class]
        return F.log_softmax(final_map, dim=1)


class ParaEncoder(nn.Module):
    def __init__(self, config, word_encoder, sent_encoder):

        super(ParaEncoder, self).__init__()
        self.config = config
        self.word_encoder = word_encoder
        self.sent_encoder = sent_encoder
        # self.pooling = config.pooling
        self.para_pooling = config.para_pooling
        self.n_classes = config.class_number
        self.sent_gru_hidden = sent_encoder.sent_gru_hidden
        self.para_gru_hidden = config.para_gru_hidden
        self.dropout = nn.Dropout(config.dropout)
        self.encoder_type = config.encoder_type
        if self.para_pooling == 'attn':
            self.ws1 = nn.Linear(2 * self.sent_gru_hidden, 2 * self.sent_gru_hidden, bias=True)
            self.ws2 = nn.Linear(2 * self.sent_gru_hidden, 1, bias=False)
            self.tanh = nn.Tanh()
            self.ws1.weight.data.uniform_(-0.1, 0.1)
            self.ws2.weight.data.uniform_(-0.1, 0.1)
        if config.encoder_type == 'LSTM':
            self.para_encoder = nn.LSTM(2 * self.sent_gru_hidden, self.para_gru_hidden, bidirectional=True)
        elif config.encoder_type == 'GRU':
            self.para_encoder = nn.GRU(2 * self.sent_gru_hidden, self.para_gru_hidden, bidirectional=True)
        self.final_linear = nn.Linear(2 * self.para_gru_hidden, self.n_classes)

    def forward(self, mini_batch):
        max_sents, batch_size, max_tokens = mini_batch.size()
        state_word = self.word_encoder.init_hidden(batch_size)
        state_sent = self.sent_encoder.init_hidden(batch_size)
        state_para = self.init_hidden()
        s = None
        if self.config.word_pooling == 'attn':
            for i in range(max_sents):
                _s, state_word, _ = self.word_encoder(mini_batch[i, :, :].transpose(0, 1), state_word)
                if (s is None):
                    s = _s
                else:
                    s = torch.cat((s, _s), 0)
        else:
            for i in range(max_sents):
                _s, state_word = self.word_encoder(mini_batch[i, :, :].transpose(0, 1), state_word)
                if (s is None):
                    s = _s
                else:
                    s = torch.cat((s, _s), 0)
        if self.config.sent_pooling == 'attn': _, _, sent_vectors = self.sent_encoder(batch_size, s, state_sent)  # sent_vectors: [plen, nhid*2]
        else: sent_vectors = self.sent_encoder(batch_size, s, state_sent)  # sent_vectors: [plen, nhid*2]

        output_para, state_para = self.para_encoder(self.dropout(sent_vectors.unsqueeze(1)), state_para) # [plen, 1, nhid*2]

        if self.para_pooling == 'mean':
            para_vectors = torch.mean(output_para, 0)
        elif self.para_pooling == 'max':
            para_vectors = torch.max(output_para, 0)[0] # [1, 2*hid]
        elif self.para_pooling == 'attn':
            output_para, state_para = self.para_encoder(self.dropout(sent_vectors.unsqueeze(1)),
                                                        state_para)  # [plen, 1, nhid*2]
            para_squish = self.tanh(self.ws1(self.dropout(output_para)))  # [plen, 1, nhid*2]
            para_attn = self.ws2(para_squish)  # [plen, 1, 1]
            para_attn_norm = softmax(para_attn, dim=0)  # [plen, 1, 1]
            # print(output_para.shape, para_attn_norm.shape)
            para_vectors = torch.mm(para_attn_norm.squeeze(2).transpose(0, 1),
                                         output_para.squeeze(1))  # [1, nhid*2]

        final_map = self.final_linear(para_vectors) # [1, n_class]
        return F.log_softmax(final_map, dim=1)

    def init_hidden(self):
        if self.encoder_type == 'GRU':
            return Variable(torch.zeros(2, 1, self.para_gru_hidden)).cuda()
        elif self.encoder_type == 'LSTM':
            return (Variable(torch.zeros(2, 1, self.para_gru_hidden)).cuda(),
                    Variable(torch.zeros(2, 1, self.para_gru_hidden)).cuda())


class ConvolutionalParagraphAttention(nn.Module):
    def __init__(self, config, word_model):
        super(ConvolutionalParagraphAttention, self).__init__()
        self.config = config
        self.word_model = word_model
        self.para_gru_hidden = config.para_gru_hidden
        self.attention_hidden = self.para_gru_hidden
        if config.word_model.lower() == "cnn":
            self.sent_hidden = len(word_model.kernel_sizes) * word_model.kernel_num
        else:
            self.sent_hidden = 2 * word_model.word_gru_hidden
        self.n_classes = config.class_number
        self.dropout = nn.Dropout(config.dropout)
        self.para_pooling = config.para_pooling
        self.encoder_type = config.encoder_type

        if config.encoder_type == 'LSTM':
            self.para_encoder = nn.LSTM(self.sent_hidden, self.para_gru_hidden, bidirectional=True)
        elif config.encoder_type == 'GRU':
            self.para_encoder = nn.GRU(self.sent_hidden, self.para_gru_hidden, bidirectional=True)

        self.ws1 = nn.Linear(2 * self.para_gru_hidden, 2 * self.attention_hidden, bias=True)
        self.ws2 = nn.Linear(2 * self.attention_hidden, 1, bias=False)
        self.final_linear = nn.Linear(2 * self.para_gru_hidden, self.n_classes)
        self.tanh = nn.Tanh()
        self.ws1.weight.data.uniform_(-0.1, 0.1)
        self.ws2.weight.data.uniform_(-0.1, 0.1)

    def forward(self, mini_batch):
        logits, hidden = self.word_model.forward(mini_batch) # hidden [bsz(plen), 300]
        state_para = self.init_hidden()
        output_para, state_para = self.para_encoder(self.dropout(hidden.unsqueeze(1)), state_para)  # [plen, 1, nhid*2]

        if self.para_pooling == 'attn':
            if mini_batch.shape[1] == 1:
                para_attn_norm = torch.tensor([[[1.0]]])
                para_vectors = output_para.squeeze(1) # [plen, nhid*2]
            else:
                para_squish = self.tanh(self.ws1(self.dropout(output_para)))  # [plen, 1, nhid*2]
                para_attn = self.ws2(para_squish)  # [plen, 1, 1]
                para_attn_norm = softmax(para_attn, dim=0)  # [plen, 1, 1]
                # print(output_para.shape, para_attn_norm.shape)
                para_vectors = torch.mm(para_attn_norm.squeeze(2).transpose(0, 1), output_para.squeeze(1))  # [plen, nhid*2]
            final_map = self.final_linear(para_vectors)  # [1, n_class]
            return F.log_softmax(final_map, dim=1), para_attn_norm

        elif self.para_pooling == 'ensem-attn':
            if mini_batch.shape[1] == 1:
                return F.log_softmax(logits, dim=1)
            y_pred = F.softmax(logits, dim=1) # [1, plen]
            para_squish = self.tanh(self.ws1(self.dropout(output_para.squeeze(1))))  # [plen, nhid*2]
            # print(para_squish.shape)
            para_attn = self.ws2(para_squish)  # [plen, 1]
            para_attn_norm = softmax(para_attn, dim=0)  # [plen, 1]
            doc_pred = torch.mm(para_attn_norm.transpose(0, 1), y_pred)  # [1, plen]
            # print(doc_pred)
            return torch.log(doc_pred)

        elif self.para_pooling == 'mean':
            para_vectors = torch.mean(output_para, 0)
        elif self.para_pooling == 'max':
            para_vectors = torch.max(output_para, 0)[0] # [1, 2*hid]
        final_map = self.final_linear(para_vectors)  # [1, n_class]
        return F.log_softmax(final_map, dim=1)

    def init_hidden(self):
        if self.encoder_type == 'GRU':
            return Variable(torch.zeros(2, 1, self.para_gru_hidden)).cuda()
        elif self.encoder_type == 'LSTM':
            return (Variable(torch.zeros(2, 1, self.para_gru_hidden)).cuda(),
                    Variable(torch.zeros(2, 1, self.para_gru_hidden)).cuda())