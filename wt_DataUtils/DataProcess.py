from DataLoad.DataLoad_NER import *
import os
from wt_DataUtils.Alphabet import CreateAlphabet
import time
import shutil
from wt_DataUtils.batch_iterator import *
from wt_DataUtils.UniversalData import *
from wt_DataUtils.wt_Embed import *
from model.Sequence_Label_Model import *
from model_test import *


def get_learning_algorithm(config):
    algorithm = None
    if config.adam is True:
        algorithm = "Adam"
    elif config.sgd is True:
        algorithm = "SGD"
    print("learning algorithm is {}.".format(algorithm))
    return algorithm


def get_params(config, alphabet):
    """
    :param config:
    :param alphabet:
    :return:
    """
    # get algorithm
    config.learning_algorithm = get_learning_algorithm(config)

    # save the best model path
    config.save_best_model_path = config.save_best_model_dir
    if config.test is False:
        if os.path.exists(config.save_best_model_path):
            shutil.rmtree(config.save_best_model_path)

    # get params
    config.embed_num = alphabet.word_alphabet.vocab_size
    config.char_embed_num = alphabet.char_alphabet.vocab_size
    config.class_num = alphabet.label_alphabet.vocab_size
    config.paddingId = alphabet.word_paddingId
    config.char_paddingId = alphabet.char_paddingId
    config.label_paddingId = alphabet.label_paddingId
    config.create_alphabet = alphabet
    print("embed_num {}, char_embed_num {}, class_num {}".format(config.embed_num, config.char_embed_num, config.class_num))

    print("PaddingID {}".format(config.paddingId))
    print("char paddingID {}".format(config.char_paddingId))


def save_dict2file(dict, path):
    """
    :param dict:  dict
    :param path:  path that it is used to save dict
    :return:
    """
    print("Save dictionary")
    if os.path.exists(path):
        print("path {} is exist, deleted".format(path))
    file = open(path, encoding="UTF-8", mode='w')
    for word, inedx in dict.items():
        file.write(str(word) + "\t" + str(inedx) + "\n")
    file.close()
    print("Saving dictionary is finshed.")


def ProcessingData(config):
    """
    :param config:
    :return:
    """
    print("processing data......")
    load_data = DataLoad(path=[config.train_file, config.dev_file, config.test_file], shuffle=True, config=config)
    train_data, dev_data, test_data = load_data.dataLoader()  # 这一步已经将train ,dev,test中的数据处理完成。
    print("train_sentence:{}, dev_sentence:{}, test_sentence:{}.".format(len(train_data),  len(dev_data), len(test_data)))
    data_dict = {"train_data": train_data, "dev_data": dev_data, "test_data": test_data}
    if config.save_pkl:
        torch.save(obj=data_dict, f=os.path.join(config.pkl_directory, config.pkl_data))

    alphabet = None
    alphabet = CreateAlphabet(min_freq=config.min_freq, train_data=train_data, dev_data=dev_data, test_data=test_data,
                             config=config)
    alphabet.build_vocab()
    alpabet_dict = {"alphabet": alphabet}
    if config.save_pkl:
        torch.save(obj=alpabet_dict, f=os.path.join(config.pkl_directory, config.pkl_alphabet))

    # Create iterator
    create_iter = Iterators(batch_size=[config.batch_size, config.dev_batch_size, config.test_batch_size],
                            data=[train_data, dev_data, test_data], alphabet=alphabet, config=config)
    train_iter, dev_iter, test_iter = create_iter.Createiterator()
    iter_dict = {"train_iter": train_iter, "dev_iter": dev_iter, "test_iter": test_iter}
    if config.save_pkl:
        torch.save(obj=iter_dict, f=os.path.join(config.pkl_directory, config.pkl_iter))

    return train_iter, dev_iter, test_iter, alphabet


def pre_embed(config, alphabet):
    print("外部预训练打方式选择........................")
    pretrain_embed = None
    embed_types = ""
    if config.pretrained_embed and config.zeros:
        embed_types = "zeros"
    elif config.pretrained_embed and config.avg:
        embed_types = "avg"
    elif config.pretrained_embed and config.uniform:
        embed_types = "uniform"
    elif config.pretrained_embed and config.nnembed:
        embed_types = "nn"
    if config.pretrained_embed is True:
        p = Embed(path=config.pretrained_embed_file, words_dict=alphabet.word_alphabet.id2words,
                  embed_type=embed_types, pad=paddingkey)
        pretrain_embed = p.get_embed()

        embed_dict = {"pretrain_embed": pretrain_embed}
        torch.save(obj=embed_dict, f=os.path.join(config.pkl_directory, config.pkl_embed))

    return pretrain_embed


def load_model(config):
    """
    :param config:
    :return:  select model
    """
    print("Select model for training......")
    model = Sequence_Label_Model(config)
    print("Save model to  {}".format(config.save_dir))
    shutil.copytree("model", "/".join([config.save_dir, "model"]))
    if config.device != cpu_device:
        model = model.cuda()
    if config.test is True:
        model = load_test_model(model, config)
    print(model)
    return model


def LoadData(config):
    """
    :return:
    """
    print("get data for process or pkl data........")
    train_iter, dev_iter, test_iter = None, None, None
    alphabet = None
    start_time = time.time()
    if (config.train is True) and (config.process is True):
        print("process data......")
        if os.path.exists(config.pkl_directory): shutil.rmtree(config.pkl_directory)
        # shutil.rmtree()表示递归删除文件夹下的所有子文件夹和子文件。
        if not os.path.isdir(config.pkl_directory):os.makedirs(config.pkl_directory)
        # os.path.isdir()函数判断某一路径是否为目录
        train_iter, dev_iter, test_iter, alphabet = ProcessingData(config)
        config.pretrained_weight = pre_embed(config=config, alphabet=alphabet)
    elif ((config.train is True) and (config.process is False)) or (config.test is True):
        print("load data from pkl file")
        alphabet_dict = torch.load(os.path.join(config.pkl_directory, config.pkl_alphabet))
        print(alphabet_dict.keys())
        alphabet = alphabet_dict["alphabet"]
        iter_dict = torch.load(f=os.path.join(config.pkl_directory, config.pkl_iter))
        print(iter_dict.keys())
        train_iter, dev_iter, test_iter = iter_dict.values()
        config.pretrained_weight = None
        if os.path.exists(os.path.join(config.pkl_directory, config.pkl_embed)):
            embed_dict = torch.load(f=os.path.join(config.pkl_directory, config.pkl_embed))
            print(embed_dict.keys())
            embed = embed_dict["pretrain_embed"]
            config.pretrained_weight = embed
    end_time = time.time()
    print("All data/Alphabet/Iterator Use Time {:.4f}".format(end_time - start_time))
    print("******************************************")
    return train_iter, dev_iter, test_iter, alphabet







