from torch.autograd import Variable
import torch


class Batch_Features:
    def __init__(self):
        self.batch_length = 0
        self.batch = None
        self.word_features = 0
        self.char_features = 0
        self.label_features = 0
        self.Origin_sentence_length = 0

    @staticmethod
    def cuda(features):
        features.word_features = features.word_features.cuda()
        features.char_features = features.char_features.cuda()
        features.label_features = features.label_features.cuda()


class Iterators:
    def __init__(self, batch_size=None, data=None, alphabet=None, config=None):
        self.config = config
        self.batch_size = batch_size
        self.data = data
        self.alphabet = alphabet
        self.alphabet_static = None
        self.iterator = []
        self.batch = []
        self.features = []
        self.data_iter = []
        self.max_char_len = config.max_char_len

    def Createiterator(self):
        """
        :param:data [train_data, dev_data, test_data]
        :return:
        """
        assert isinstance(self.data, list), "Error:data must be in list[train_data, dev_data, test_data]."
        assert isinstance(self.batch_size, list), "Error:batch_size should look like this [16,16,2]."
        for id_data in range(len(self.data)):
            print("******************* Now start building {} iterator ******************".format(id_data + 1))
            self.convert_word2id(self.data[id_data], self.alphabet)  # 将train，dev,test中的每个单词都转化成对应的id
            self.features = self.Create_Each_Iterator(insts=self.data[id_data], batch_size=self.batch_size[id_data],
                                      alphabet=self.alphabet)
            self.data_iter.append(self.features)
            self.features = []
        if len(self.data_iter) == 2:
            return self.data_iter[0], self.data_iter[1]
        if len(self.data_iter) == 3:
            return self.data_iter[0], self.data_iter[1], self.data_iter[2]

    @staticmethod
    def convert_word2id(insts, alphabet):
        """
        :param insts: self.data[id_data]
        :param alphabet:
        :return:
        """
        for inst in insts:
            for index in range(inst.words_size):
                # word
                word = inst.words[index]
                wordId = alphabet.word_alphabet.from_string(word)
                if wordId == -1:
                    wordId = alphabet.word_unkId
                inst.words_index.append(wordId)

                # label
                label = inst.labels[index]
                labelId = alphabet.label_alphabet.from_string(label)
                inst.labels_index.append(labelId)

                char_index = []
                for char in inst.chars[index]:
                    charId = alphabet.char_alphabet.from_string(char)
                    if charId == -1:
                        charId = alphabet.char_unkId
                    char_index.append(charId)
                inst.chars_index.append(char_index)

    def Create_Each_Iterator(self, insts, batch_size, alphabet):
        """
        :param insts: data[id_data]   train    dev     test
        :param batch_size:
        :param alphabet:
        :return:
        """
        batch = []
        count_inst = 0
        for index, inst in enumerate(insts):
            batch.append(inst)
            count_inst += 1
            if len(batch) == batch_size or count_inst == len(insts):
                one_batch = self._Create_Each_Batch(batch=batch, batch_size=batch_size, alphabet=alphabet)
                self.features.append(one_batch)
                batch = []
        print("The all data has created iterator.")
        return self.features

    def _Create_Each_Batch(self, batch, batch_size, alphabet):
        """
        :param batch: one batch list
        :param batch_size:
        :param alphabet:
        :return:
        """
        batch_length = len(batch)
        max_word_size = -1
        max_label_size = -1
        Orig_sentence_length = []
        for inst in batch:
            Orig_sentence_length.append(inst.words_size)
            word_size = inst.words_size
            if word_size > max_word_size:
                max_word_size = word_size

            if len(inst.labels) > max_label_size:  # ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
                # 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'S-ORG', 'O', 'O', 'O', 'B-PER', 'I-PER', 'I-PER', 'E-PER', 'O',]
                max_label_size = len(inst.labels)
        assert max_word_size == max_label_size  # 保证一个词对应一个标签

        batch_word_features = Variable(torch.zeros(batch_length, max_word_size).type(torch.LongTensor))
        batch_char_features = Variable(torch.zeros(batch_length, max_word_size, self.max_char_len).type(torch.LongTensor))
        batch_label_features = Variable(torch.zeros(batch_length * max_word_size).type(torch.LongTensor))
        for id_inst in range(batch_length):
            inst = batch[id_inst]
            for id_word_index in range(max_word_size):
                # Word Feature

                if id_word_index < inst.words_size:
                    batch_word_features.data[id_inst][id_word_index] = inst.words_index[id_word_index]
                else:
                    batch_word_features.data[id_inst][id_word_index] = alphabet.word_paddingId

                # Label Feature

                if id_word_index < len(inst.labels_index):
                    batch_label_features.data[id_inst * max_word_size + id_word_index] = inst.labels_index[id_word_index]
                else:
                    batch_label_features.data[id_inst * id_word_index + id_word_index] = alphabet.label_paddingId

                # Char Features

                word_char_size = len(inst.chars_index[id_word_index]) if id_word_index < inst.words_size else 0
                for id_word_c in range(self.max_char_len):
                    if id_word_c < word_char_size:
                        batch_char_features.data[id_inst][id_word_index][id_word_c] = inst.chars_index[id_word_index][id_word_c]
                    else:
                        batch_char_features.data[id_inst][id_word_index][id_word_c] = alphabet.char_paddingId

        features = Batch_Features()
        features.batch_length = batch_length
        features.batch = batch
        features.word_features = batch_word_features
        features.char_features = batch_char_features
        features.label_features = batch_label_features
        features.Origin_sentence_length = Orig_sentence_length

        if self.config.use_cuda is True:
            features.cuda(features)
        return features





