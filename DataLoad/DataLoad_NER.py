from DataLoad.Instance import *
from wt_DataUtils.UniversalData import char_pad
import torch
import re
import os
torch.manual_seed(seed_num)
random.seed(seed_num)


class DataLoadHelp(object):

    @staticmethod
    def _clean_str(string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
        """
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    @staticmethod
    def _write_shuffle_datasets_to_file(insts, path):
        """
        :param insts:
        :param path:
        :return:
        """
        w_path = ".".join([path, shuffle])
        if os.path.exists(w_path):
            os.remove(w_path)
        file = open(w_path, encoding="utf-8", mode="w")
        for id, inst in enumerate(insts):
            for word, label in zip(inst.words, inst.labels):
                file.write(" ".join([word, label, '\n']))
            file.write("\n")
        print("Write shuffle dataset to file {}".format(w_path))


class DataLoad(DataLoadHelp):
    def __init__(self, path, shuffle, config):
        """
        :param path: path list
        :param shuffle:
        :param config:
        """
        self.data_list = []
        self.max_count = config.max_count
        self.path = path
        self.shuffle = shuffle
        self.pad_char = [char_pad, char_pad]
        self.max_char_len = config.max_char_len

    def dataLoader(self):
        path = self.path
        shuffle = self.shuffle
        assert isinstance(path, list), "path must be list"
        print("output path {}:".format(path))
        for id_data in range(len(path)):
            print("At this time, we are dealing with:：{}".format(path[id_data]))
            insts = self.load_Each_Data(path=path[id_data], shuffle=shuffle)
            random.shuffle(insts)  # train, dev, test sets 全部打乱
            self._write_shuffle_datasets_to_file(insts, path=path[id_data])
            if shuffle is True:
                print("shuffle data......")
                random.shuffle(insts)
            # 不做排序处理
            self.data_list.append(insts)
        if len(self.data_list) == 3:
            return self.data_list[0], self.data_list[1], self.data_list[2]
        elif len(self.data_list) == 2:
            return self.data_list[0], self.data_list[1]

    def load_Each_Data(self, path=None, shuffle=False):
        """
        :param path: path[id_data]
        :param shuffle:
        :return:
        """
        assert path is not None, "path can't be Empty! "
        insts = []
        with open(path, encoding="UTF-8") as f:
            inst = Instance()  # 一句话建立一个实例
            for line in f.readlines():
                line = line.strip()
                if line == "" and len(inst.words) != 0:
                    inst.words_size = len(inst.words)
                    insts.append(inst)
                    inst = Instance()
                else:
                    line = line.split(" ")  # 把单词和标签以逗号分开。
                    word = line[0]
                    char = self.word_to_char(word)
                    word = self.normalize_word(word)
                    inst.chars.append(char)  # 一句话中的每个单词的字符在各自的列表里放着，同时这些列表存放在chars大列表中。
                    inst.words.append(word)
                    inst.labels.append(line[-1])
                if len(insts) == self.max_count:
                    break
            if len(inst.words) != 0:
                inst.words_size = len(inst.words)
                insts.append(inst)

        return insts

    @staticmethod
    def normalize_word(word):
        """
        :param word: 每行中的单词
        :return:
        """
        new_word = ""
        for char in word:
            if char.isdigit():
                new_word += '0 '
            else:
                new_word += char
        return new_word

    def word_to_char(self, word):
        """
        :param word: path[id_data]中的每行的单词
        :return:
        """
        char = []  
        for i in range(len(word)):
            char.append(word[i])
        if len(char) > self.max_char_len:
            half = self.max_char_len//2
            final_char = word[:half] + word[-(self.max_char_len - half):]
            char = final_char
        else:
            for i in range(self.max_char_len - len(char)):
                char.append(char_pad)
        return char

