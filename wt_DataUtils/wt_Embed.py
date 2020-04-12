from collections import OrderedDict
import tqdm
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import torch


class Embed(object):
    """
    Embed
    :param path config.pretrained_embed_file
    """
    def __init__(self, path, words_dict, embed_type, pad):
        self.embed_type_list = ["zero", "avg", "uniform", "nn"]
        self.path = path
        self.words_dict = words_dict
        self.embed_type = embed_type
        self.pad = pad

        # print(self.words_dict)
        if not isinstance(self.words_dict, dict):  # isinstance() 函数来判断一个对象是否是一个已知的类型
            self.words_dict, self.words_list = self.list2dict(self.words_dict)
        if pad is not None: self.padID = self.words_dict[pad]
        self.dim, self.words_count = self._get_dim(path=self.path), len(self.words_dict)
        self.exact_count, self.fuzzy_count, self.oov_count = 0, 0, 0

    def _get_dim(self, path):
        path = self.path
        embedding_dim = -1
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip().split(' ')
                if len(line) == 1:
                    embedding_dim = line[0]
                elif len(line) == 2:
                    embedding_dim = line[1]
                    break
                else:
                    embedding_dim = len(line) - 1
                    break
        return embedding_dim

    def get_embed(self):
        """
        :return:
        """
        embed_dict = None
        if self.embed_type in self.embed_type_list:
            embed_dict = self._read_file(path=self.path)
        else:
            print("embed_type is illegal, must be in {}".format(self.embed_type_list))
            exit()
        embed = None
        if self.embed_type == 'nn':
            embed = self._nn_embed(embed_dict=embed_dict, words_dict=self.words_dict)
        elif self.embed_type == 'zero':
            embed = self._zero_embed(embed_dict, self.words_dict)
        elif self.embed_type == 'uniform':
            embed = self._uniform_embed(embed_dict=embed_dict, words_dict=self.words_dict)
        elif self.embed_type == 'avg':
            embed = self._avg_embed(embed_dict=embed_dict, words_dict=self.words_dict)
        self.info()
        return embed

    def _nn_embed(self, embed_dict, words_dict):
        """
        :param embed_dict:
        :param words_dict:
        :return:
        """
        print("Setting pre_train word embedding by nn.Embedding for word that is out of vocabulary......")
        embed = nn.Embedding(int(self.words_count), int(self.dim))
        init.xavier_uniform_(embed.weight.data)
        embeddings = np.array(embed.weight.data)
        for word in words_dict:
            if word in embed_dict:
                embeddings[words_dict[word]] = np.array([float(i) for i in embed_dict[word]], dtype='float32')
                self.exact_count += 1
            elif word.lower() in embed_dict:
                embeddings[words_dict[word]] = np.array([float(i) for i in embed_dict[word.lower()]], dtype='float32')
                self.fuzzy_count += 1
            else:
                self.oov_count += 1
        embeddings[self.padID] = 0
        final_embed = torch.from_numpy(embeddings).float()
        return final_embed

    def _zero_embed(self, words_dict, embed_dict ):
        print("Setting pre_train word embedding by zeros for word that is out of vocabulary......")
        embeddings = np.zeros(int(self.words_count), int(self.dim))
        for word in words_dict:
            if word in embed_dict:
                embeddings[words_dict[word]] = np.array([float(i) for i in embed_dict[word]], dtype='float32')
                self.exact_count += 1
            elif word.lower() in embed_dict:
                embeddings[words_dict[word]] = np.array([float(i) for i in embed_dict[word.lower()]])
                self.fuzzy_count  += 1
            else:
                self.oov_count += 1
        final_embed = torch.from_numpy(embeddings).float()
        return final_embed

    def _uniform_embed(self, words_dict, embed_dict):
        print("Setting pre_train word embedding by uniform for word that is out of vocabulary......")
        embeddings = np.zeros(int(self.words_count), int(self.dim))  # 注意：pad 一开始就初始化为0了，所以后边不需要进行操作了。
        inword_list = {}
        for word in words_dict:
            if word in embed_dict:
                embeddings[words_dict[word]] = np.array([float(i) for i in embed_dict[word]], dtype='float32')
                self.exact_count += 1
            elif word.lower() in embed_dict:
                embeddings[words_dict[word]] = np.array([float(i) for i in embed_dict[word.lower()]], dtype='float32')
                inword_list[words_dict[word]] = 1
                self.fuzzy_count += 1
            else:
                self.oov_count += 1
        uniform_col = np.random.uniform(-0.25, 0.25, int(self.dim)).round(6)
        for i in range(int(self.words_count)):
            if i not in inword_list and i != self.padID:
                embeddings[i] = uniform_col
        final_embed = torch.from_numpy(embeddings).float()
        return final_embed

    def _avg_embed(self, words_dict, embed_dict):
        print("Setting pre_train word embedding by avg for word that is out of vocabulary......")
        embeddings = np.zeros(int(self.words_count), int(self.dim))
        inword_list = {}
        for word in words_dict:
            if word in embed_dict:
                embeddings[words_dict[word]] = np.array([float(i) for i in embed_dict[word]], dtype='float32')
                inword_list[words_dict[word]] = 1
                self.exact_count += 1
            elif word.lower() in embed_dict:
                embeddings[words_dict[word]] = np.array([float(i) for i in embed_dict[word.lower()]], dtype='float32')
                inword_list[words_dict[word]] = 1
                self.fuzzy_count += 1
            else:
                self.oov_count += 1
        sum_col = np.sum(embeddings, axis=0)/len(inword_list)
        for i in range(int(self.words_count)):
            if i not in inword_list and i != self.padID:
                embeddings[i] = sum_col
        final_embed = torch.from_numpy(embeddings).float()
        return final_embed

    def info(self):
        """
        :return:
        """
        total_count = self.exact_count + self.fuzzy_count
        print("words count {}, Embed dim {}".format(self.words_count, self.dim))
        print("Exact count {} / {}".format(self.exact_count, self.words_count))
        print("fuzzy count {} / {}".format(self.fuzzy_count, self.words_count))
        print("INV count {} / {}".format(total_count, self.words_count))
        print("OOV count {} / {}".format(self.oov_count, self.words_count))
        print("OOV proportion -----> {}".format(np.round((self.oov_count / self.words_count) * 100, 2)))
        print(40 * "*")

    @staticmethod
    def _read_file(path):
        """
        :param path: embed file path
        :return:
        """
        embed_dict = {}
        with open(path,encoding="utf-8") as f:
            lines = f.readlines()
            lines = tqdm.tqdm(lines)
            for line in lines:
                values = line.strip().split(' ')
                if len(values) == 1 or len(values) == 2 or len(values) == 3:
                    continue
                w, d = values[0], values[1:]
                embed_dict[w] = d
        return embed_dict

    @staticmethod
    def list2dict(convert_list):
        """
        :param convert_list: self.words_dict
        :return:
        """
        list_dict = OrderedDict()
        list_lower = []
        for index, word in enumerate(convert_list):
            list_lower.append(word.lower())
            list_dict[word] = index
        assert len(list_lower) == len(list_dict)
        return list_dict, list_lower





    