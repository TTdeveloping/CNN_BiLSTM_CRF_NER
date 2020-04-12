from wt_DataUtils.UniversalData import *
import collections


class CreateAlphabet:
    def __init__(self, min_freq=1, train_data=None, dev_data=None, test_data=None, config=None):
        self.min_freq = min_freq
        self.config = config
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data

        self.word_dict = collections.OrderedDict()
        self.label_dict = collections.OrderedDict()
        self.char_dict = collections.OrderedDict()

        self.word_dict[unkkey] = self.min_freq
        self.word_dict[paddingkey] = self.min_freq
        self.char_dict[unkkey] = self.min_freq
        self.char_dict[paddingkey] = self.min_freq
        self.label_dict[paddingkey] = 1

        self.word_alphabet = Alphabet(min_freq=self.min_freq)
        self.char_alphabet = Alphabet(min_freq=self.min_freq)
        self.label_alphabet = Alphabet()

        # unk key
        self.word_unkId = 1
        self.char_unkId = 1

        # paddind key
        self.word_paddingId = 0
        self.char_paddingId = 0
        self.label_paddingId = 0

    @staticmethod
    def build_data(train_data=None, dev_data=None, test_data=None):
        """
        :param train_data:
        :param dev_data:
        :param test_data:
        :return:
        """
        assert train_data is not None, "The train data can be not empty"
        datasets = []
        datasets.extend(train_data)  # extend()函数，用于扩充原来的列表。将括号中的列表的内容放在.前面的列表中，
        # 最终形成一个新的列表
        print("The length of train data is :{}".format(len(datasets)))
        if dev_data is not None:
            print("The length of dev data is :{}".format(len(dev_data)))
            datasets.extend(dev_data)
        if test_data is not None:
            print("The length of test data is :{}".format(len(test_data)))
            datasets.extend(test_data)
        print("The length of data Create Alphabet is :{}".format(len(datasets)))
        return datasets

    def build_vocab(self):
        train_data = self.train_data
        dev_data = self.dev_data
        test_data = self.test_data
        print("Start building vocab.......... ")
        datasets = self.build_data(train_data=train_data, dev_data=dev_data, test_data=test_data)
        # train,dev,test的内容都放在datasets列表中。
        for index, data in enumerate(datasets):
            # word
            for word in data.words:
                if word not in self.word_dict:
                    self.word_dict[word] = 1
                else:
                    self.word_dict[word] += 1

            # char
            for word_char in data.chars:  # 特别注意：data.chars中存放的是一个个单词的字符，每个单词的字符存在自己的列表中。
                for char in word_char:  # 遍历每个单词中的字符
                    if char.isalnum() is False:
                        continue
                    if char not in self.char_dict:
                        self.char_dict[char] = 1
                    else:
                        self.char_dict[char] += 1

            # label
            for label in data.labels:
                if label not in self.label_dict:
                    self.label_dict[label] = 1
                else:
                    self.label_dict[label] += 1

        # initial_Word2IdAndId2word()作用：就是将大于等于词频的单词，字符，标签放入函数UpdateWord2IdAndId2word()中返回它们的id

        # Create id_to_word and word_to_id by Class Alphabet
        self.word_alphabet.initial_Word2IdAndId2word(self.word_dict)
        # self.word_dict里存放的是单词和对应的统计次数。
        self.char_alphabet.initial_Word2IdAndId2word(self.char_dict)
        # self.char_dict存放字符和它对应的统计次数。
        self.label_alphabet.initial_Word2IdAndId2word(self.label_dict)
        # self.label_dict存放标签和它对应的统计次数。

        # unkId
        self.word_unkId = self.word_alphabet.from_string(unkkey)
        self.char_unkId = self.char_alphabet.from_string(unkkey)
        # paddingId
        self.word_paddingId = self.word_alphabet.from_string(paddingkey)
        self.char_paddingId = self.char_alphabet.from_string(paddingkey)
        self.label_paddingId = self.label_alphabet.from_string(paddingkey)
        # fix the vocab
        self.word_alphabet.set_fixed_flag(True)
        self.label_alphabet.set_fixed_flag(True)


class Alphabet:
    def __init__(self, min_freq=1):
        self.id2words = []
        self.words2id = collections.OrderedDict()
        self.vocab_size = 0
        self.min_freq = min_freq
        self.max_cap = 1e8
        self.fixed_vocab = False

    def initial_Word2IdAndId2word(self, data):
        """
        :param data:self.word_dict;
                    self.char_dict;
                    self.label_dict
        :return:
        """
        for key in data:
            if data[key] >= self.min_freq:
                self.from_string(key)
        self.set_fixed_flag(True)

    def set_fixed_flag(self, bfxed):
        self.fixed_vocab = bfxed
        if (not self.fixed_vocab) and (self.vocab_size >= self.max_cap):
            self.fixed_vocab = True

    def from_string(self, key):
        if key in self.words2id:
            return self.words2id[key]
        else:
            if not self.fixed_vocab:
                newid = self.vocab_size
                self.id2words.append(key)
                self.words2id[key] = newid
                self.vocab_size += 1
                if self.vocab_size >= self.max_cap:
                    self.fixed_vocab = True
                return newid
            else:
                return -1

    def from_id(self, qid, defineStr=""):
        """
        :param qid:
        :param defineStr:
        :return:
        """
        if int(qid) < 0 or self.vocab_size <= qid:
            return defineStr
        else:
            return self.id2words[qid]







