from model.BiLSTM_CNN import *
from model.BiLSTM import *
from model.CRF import CRF


class Sequence_Label_Model(nn.Module):
    """
    Sequence Label Task
    """
    def __init__(self, config):
        super(Sequence_Label_Model, self).__init__()
        self.config = config
        # embedding need
        self.embed_num = config.embed_num
        self.embed_dim = config.embed_dim
        self.label_num = config.class_num
        self.paddingId = config.paddingId

        # dropout
        self.dropout_emb = config.dropout_emb
        self.dropout = config.dropout

        # lstm
        self.lstm_hiddens = config.lstm_hiddens
        self.lstm_layers = config.lstm_layers

        # pretrain
        self.pretrained_embed = config.pretrained_embed
        self.pretrained_weight = config.pretrained_weight

        # char
        self.use_char = config.use_char
        self.char_embed_num = config.char_embed_num
        self.char_paddingId = config.char_paddingId
        self.char_dim = config.char_dim
        self.conv_filter_sizes = self._conv_filter(config.conv_filter_sizes)
        # self.conv_filter_nums = config.conv_filter_nums
        self.conv_filter_nums = self._conv_filter(config.conv_filter_nums)

        # use crf
        self.use_crf = config.use_crf

        # cuda or cpu
        self.device = config.device

        self.target_size = self.label_num if self.use_crf is False else self.label_num + 2  # start and end
        if self.use_char is True:
            self.encoder_model = BiLSTM_CNN(embed_num=self.embed_num, embed_dim=self.embed_dim,
                                            label_num=self.target_size, paddingId=self.paddingId,
                                            dropout_emb=self.dropout_emb, dropout=self.dropout,
                                            lstm_hiddens=self.lstm_hiddens, lstm_layers=self.lstm_layers,
                                            char_paddingId=self.char_paddingId, char_embed_num=self.char_embed_num,
                                            char_dim=self.char_dim, conv_filter_sizes=self.conv_filter_sizes,
                                            conv_filter_nums=self.conv_filter_nums, pretrained_embed=self.pretrained_embed,
                                            pretrained_weight=self.pretrained_weight, device=self.device)
        else:
            self.encoder_model = BiLSTM(embed_num=self.embed_num, embed_dim=self.embed_dim, label_num=self.target_size,
                                        paddingId=self.paddingId, dropout=self.dropout, dropout_emb=self.dropout_emb,
                                        lstm_hiddens=self.lstm_hiddens, lstm_layers=self.lstm_layers,
                                        pretrained_embed=self.pretrained_embed, pretrained_weight=self.pretrained_weight,
                                        device=self.device)
        # if self.use_crf is True:
            # args_crf = dict({'target_size': self.label_num, 'device':self.device})
            # self.crf_layer = CRF(**args_crf)

    @staticmethod
    def _conv_filter(filter_height):
        """
        :return:
        """
        int_list = []
        filter_height_sizes_list = filter_height.split(",")
        for height in filter_height_sizes_list:
            int_list.append(int(height))
        return int_list

    def forward(self, word, char, sentence_length, train=False):
        if self.use_char is True:
            logit = self.encoder_model(word, char, sentence_length)
            return logit
        else:
            logit = self.encoder_model(word, sentence_length)
            return logit





