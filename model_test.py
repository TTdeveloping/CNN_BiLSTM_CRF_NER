import os
import torch
import sys
from wt_DataUtils.utils import *
from wt_DataUtils.UniversalData import *


def load_test_model(model, config):
    """
    :param model:
    :param config:
    :return:
    """
    if config.t_model is None:
        test_model_dir = config.save_best_model_dir
        test_model_name = "{}.pt".format(config.model_name)
        test_model_path = os.path.join(test_model_dir, test_model_name)
        print(" Load test model from {}".format(test_model_path))
    else:
        test_model_path = config.t_model
        print("load test model from {}".format(test_model_path))
    model.load_state_dict(torch.load(test_model_path))  # state_dict 是一个python的字典格式,以字典的格式存储,
    # 然后以字典的格式被加载,而且只加载key匹配的项 只有那些参数可以训练的layer才会被保存到模型的state_dict中,如卷积层,线性层等等
    return model


def load_test_data(train_iter, dev_iter, test_iter, config ):
    """
    :param train_iter:
    :param dev_iter:
    :param test_iter:
    :param config:
    :return:
    """
    data, shuffle_data, label_result = None, None, None
    if config.t_data is None:
        print("default[test] for model test.")
        data = test_iter
        shuffle_data_path = ".".join([config.test_file, shuffle])
        label_result_path = "{}.out".format(shuffle_data_path)
    elif config.t_data == "train":
        print("train data for model test and label.")
        data = train_iter
        shuffle_data_path = ".".join([config.train_file, shuffle])
        label_result_path = "{}.out".format(shuffle_data_path)
    elif config.t_data == "dev":
        print("dev data for model test and label.")
        data = dev_iter
        shuffle_data_path = ".".join([config.dev_file, shuffle])
        label_result_path = "{}.out".format(shuffle_data_path)
    elif config.t_data == "test":
        print("test data for model test and label.")
        data = test_iter
        shuffle_data_path = ".".join([config.test_file, shuffle])
        label_result_path = "{}.out".format(shuffle_data_path)
    else:
        print("Error value --- t_data = {}, nust be in [None, 'train', 'dev', 'test'].".format(config.t_data))
        exit()
    return data, shuffle_data_path, label_result_path


class T_Inference(object):
    """
    """
    def __init__(self, model, data, shuffle_data_path, label_result_path, alphabet, use_crf, config):
        """
        :param model:
        :param data: train_iter or dev_iter or test_iter
        :param shuffle_data_path:
        :param label_result_path:
        :param alphabet:
        :param use_crf:
        :param config:

        """
        print("Initialize T_Inference")
        self.model = model
        self.data = data
        self.shuffle_data_path = shuffle_data_path
        self.label_result_path = label_result_path
        self.alphabet = alphabet
        self.config = config
        self.use_crf = use_crf

    def infer2file(self):
        """
        :return: None

        """
        print("infer...")
        self.model.eval()
        predict_labels = []
        predict_label = []
        all_count = len(self.data)
        now_count = 0
        for data in self.data:
            now_count += 1
            sys.stdout.write("\rinfer with batch number {}/{} .".format(now_count, all_count))
            word, char, mask, sentence_length, tags = self._get_model_args(data)
            logit = self.model(word, char, sentence_length, train=False)
            if self.use_crf is False:
                predict_ids = predict_tag_id(logit)
                for id_batch in range(data.batch_length):
                    inst = data.batch[id_batch]
                    label_ids = predict_ids[id_batch]
                    for id_word in range(inst.word_size):
                        predict_label.append(self.alphabet.label_alphabet.from_id(label_ids[id_word]))

            else:
                pass
        print("\ninfer is finished. Following, we will write prediction labels to shuffle data file.")
        self.write2file(result=predict_label, shuffle_data_path=self.shuffle_data_path,

                        label_result_path=self.label_result_path)
    @staticmethod
    def write2file(result, shuffle_data_path, label_result_path):
        """
        :param result:  model prediction result
        :param shuffle_data_path:  train.txt.shuffle or dev.txt.shuffle or test.txt.shuffle
        :param label_result_path:
        :return:
        """
        print("write result to file {}".format(label_result_path))
        if os.path.exists(shuffle_data_path) is False:
            print(" shuffle data file is not exist.")
        if os.path.exists(label_result_path):
            os.remove(label_result_path)
        file_out = open(label_result_path, encoding="utf-8", mode="w")

        with open(shuffle_data_path, encoding="utf-8") as f:
            id = 0
            for line in f.readlines():
                sys.stdout.write("\rwrite with {}/{}".format(id+1, len(result)))
                if line == "\n":
                    file_out.write("\n")
                    continue
                line = line.strip().split()
                line.append(result[id])
                id += 1
                file_out.write(" ".join(line) + "\n")

                if id >= len(result):
                    break
        file_out.close()
        print("\n...*** finished.")




    @staticmethod
    def _get_model_args(batch_features):
        """
        :param batch_features:  Batch Instance
        :return:
        """
        word = batch_features.word_features
        char = batch_features.char_features
        mask = word > 0
        sentence_length = batch_features.sentence_length
        # desorted_indices = batch_features.desorted_indices
        tags = batch_features.label_features
        return word, char, mask, sentence_length, tags







