import torch.nn as nn
from model.initialize import *
from model.Modelhelp import *
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTM(nn.Module):
    """
    BiLSTM
    """
    def __init__(self, **kwargs):
        super(BiLSTM, self).__init__()
        for k in kwargs:
            self.__setattr__(k, kwargs[k])
        V = self.embed_num
        D = self.embed_dim
        C = self.label_num
        paddingId = self.paddingId

        self.embed = nn.Embedding(V, D, padding_idx=paddingId)

        if self.pretrained_embed:
            self.embed.weight.data.copy_(self.pretrained_weight)

        else:
            init_embedding(self.embed.weight)
        self.dropout_emb = nn.Dropout(self.dropout_emb)
        self.dropout = nn.Dropout(self.dropout)

        self.bilstm = nn.LSTM(input_size=D, hidden_size=self.lstm_hiddens, num_layers=self.lstm_layers,
                              bidirectional=True, batch_first=True, bias=True)

        self.linear = nn.Linear(in_features=self.lstm_hiddens * 2, out_features=C, bias=True)
        init_linear(self.linear)

    def forward(self, word, sentence_length):
        """
        :param word_embedding:
        :param sentence_length:
        :return:
        """
        word, sentence_length, disorder_indices = prepare_pack_padded_sequence(word, sentence_length, device=self.device)
        x = self.embed(word)
        x = self.dropout_emb(x)
        packed_embed = pack_padded_sequence(x, sentence_length, batch_first=True)
        x, _ = self.bilstm(packed_embed)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = x[disorder_indices]
        x = self.dropout(x)
        x = torch.tanh(x)
        logit = self.linear(x)
        return logit



