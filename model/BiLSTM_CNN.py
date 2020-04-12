import torch.nn as nn
from model.initialize import *
from wt_DataUtils.UniversalData import cpu_device


class BiLSTM_CNN(nn.Module):
    """
    BIlSTM_CNN
    """

    def __init__(self, **kwargs):
        super(BiLSTM_CNN, self).__init__()
        for k in kwargs:
            self.__setattr__(k, kwargs[k])

        V = self.embed_num
        D = self.embed_dim
        C = self.label_num
        paddingId = self.paddingId
        char_paddingId = self.char_paddingId

        # word embedding layer
        self.embed = nn.Embedding(V, D, padding_idx=paddingId)
        if self.pretrained_embed:
            self.embed.weight.data.copy_(self.pretrained_weight, self.char_dim, passing_index=self.char_paddingId)

        # char embedding layer
        self.char_embed = nn.Embedding(self.char_embed_num, self.char_dim, padding_idx=char_paddingId)
        init_embed(self.char_embed.weight)

        # dropout
        self.dropout_embed = nn.Dropout(self.dropout_emb)
        self.dropout = nn.Dropout(self.dropout)

        # cnn
        # print("jahjaha ")
        # print(self.conv_filter_sizes)
        # print(self.conv_filter_nums)
        # exit()
        self.char_encoders = []
        for i, filter_size in enumerate(self.conv_filter_sizes):
            f = nn.Conv3d(in_channels=1, out_channels=self.conv_filter_nums[i], kernel_size=(1, filter_size, self.char_dim))
            self.char_encoders.append(f)
        for conv in self.char_encoders:
            if self.device != cpu_device:
                conv.cuda()
        # bilstm
        bilstm_input_dim = D + sum(self.conv_filter_nums)
        self.bilstm = nn.LSTM(input_size=bilstm_input_dim, hidden_size=self.lstm_hiddens, num_layers=self.lstm_layers,
                              bidirectional=True, batch_first=True, bias=True)
        # fully connected layer
        self.linear = nn.Linear(in_features=self.lstm_hiddens*2, out_features=C, bias=True)
        init_linear_weight_bias(self.linear)

    def _char_forward(self, input):
        """
        :param input: 3D tensor, [bs, max_len, max_len_char]
        :return: char_conv_outputs: 3D tensor, [bs, max_len, output_dim]
        """
        max_word_len, max_char_len = input.size(1), input.size(2)
        input = input.view(-1, max_word_len * max_char_len)
        input_embed = self.char_embed(input)
        input_embed = input_embed.view(-1, 1, max_word_len, max_char_len, self.char_dim)

        # convolution char feature
        char_conv_outputs = []
        for char_encoder in self.char_encoders:
            conv_output = char_encoder(input_embed)
            pool_output = torch.squeeze(torch.max(conv_output, -2)[0], -1)
            char_conv_outputs.append(pool_output)
        char_conv_outputs = torch.cat(char_conv_outputs, dim=1)
        char_conv_outputs = char_conv_outputs.permute(0, 2, 1)

        return char_conv_outputs

    def forward(self, word, char, sentence_length):
        """
        :param word:
        :param char:
        :param sentence_length:
        :return:
        """
        char_conv = self._char_forward(char)
        char_conv = self.dropout(char_conv)
        word = self.embed(word)
        x = torch.cat((word, char_conv), -1)
        x = self.dropout_embed(x)
        x, _ = self.bilstm(x)
        x = self.dropout(x)
        x = torch.tanh(x)
        logit = self.linear(x)
        return logit


