import argparse
import Config.config as configurable
from wt_DataUtils.DataProcess import *
import random
import datetime
from training import *
from wt_DataUtils.UniversalData import *
torch.manual_seed(seed_num)
random.seed(seed_num)


def start_train(train_iter, dev_iter, test_iter, model, config):
    """
    :param train_iter:
    :param dev_iter:
    :param test_iter:
    :param model:
    :param config:
    :return:
    """
    t = Train(train_iter=train_iter, dev_iter=dev_iter, test_iter=test_iter, model=model, config=config)
    t.train()


def main():
    config.menu = datetime.datetime.now().strftime("%Y/%m/%d-%H/%M/%S")
    config.save_dir = os.path.join(config.save_direction, config.menu)
    if not os.path.isdir(config.save_dir): os.makedirs(config.save_dir)
    # get data,iter,alphabet
    train_iter, dev_iter, test_iter, alphabet = LoadData(config=config)

    # get parameters
    get_params(config=config, alphabet=alphabet)

    # save dictionary
    save_directory(config=config)

    # model
    model = load_model(config)

    if config.train is True:
        start_train(train_iter, dev_iter, test_iter, model, config)
        exit()
    elif config.test is True:
        start_test(train_iter, dev_iter, test_iter, model, alphabet, config)
        exit()


def start_test(train_iter, dev_iter, test_iter, model, alphabet, config):
    """
    :param train_iter:
    :param dev_iter:
    :param test_iter:
    :param model:
    :param alphabet:
    :param config:
    :return:
    """
    print("\nTesting Start......")
    data, shuffle_data_path, label_result_path = load_test_data(train_iter, dev_iter, test_iter, config)
    infer = T_Inference(model=model, data=data, shuffle_data_path=shuffle_data_path, label_result_path=label_result_path,
                        use_crf=config.use_crf, config=config)
    infer.infer2file()


def save_directory(config):
    """
    :param config:
    :return:
    """
    if config.save_dict is True:
        if os.path.exists(config.dict_directory):
            shutil.rmtree(config.dict_directory)
        if not os.path.isdir(config.dict_directory):
            os.makedirs(config.dict_directory)
        config.word_dict_path = "/".join([config.dict_directory, config.word_dict])
        config.label_dict_path = "/".join([config.dict_directory, config.label_dict])
        print("word_dict_path : {}".format(config.word_dict_path))
        print("label_dict_path : {}".format(config.label_dict_path))
        save_dict2file(config.create_alphabet.word_alphabet.words2id, config.word_dict_path)
        save_dict2file(config.create_alphabet.label_alphabet.words2id, config.label_dict_path)
        print("copy dictionary to {}".format(config.save_dir))
        shutil.copytree(config.dict_directory, "/".join([config.save_dir, config.dict_directory]))


def parse_print():

    parser = argparse.ArgumentParser(description="NER")
    parser.add_argument("-c", "--config", dest="config_file", type=str, default="./Config/config.cfg", help="config path")
    parser.add_argument("-device", "--device", dest="device", type=str, default="cuda:0", help="device[‘cpu’,‘cuda:0’,‘cuda:1’,......]")
    parser.add_argument("--train", dest="train", action="store_true", default=True, help="TrainModel")
    parser.add_argument("-p", "--process", dest="process", action="store_true", default=True, help="DataProcess")
    parser.add_argument("-t", "--test", dest="test", action="store_true", default=False, help="TestModel")
    parser.add_argument("--t_model", dest="t_model", type=str, default=None, help="model for test")
    parser.add_argument("--t_data", dest="t_data", type=str, default=None, help="data[train dev test]")
    parser.add_argument("--predict", dest="predict", action="store_true", default=False, help="PredictModel")
    args = parser.parse_args()
    # print(args.config_file)
    # exit()
    config = configurable.Configurable(config_file=args.config_file)
    config.device = args.device
    config.train = args.train
    config.process = args.process
    config.test = args.test
    config.t_model = args.t_model
    config.t_data = args.t_data
    config.predict = args.predict

    if config.test is True:
        config.train = False
    if config.t_data not in [None, "train", "dev", "test"]:
        print("\nUsage")
        parser.print_help()
        print("t_data : {}, not in [None, 'train, 'dev', 'test']".format(config.t_data))
        exit()
    print("+++++++++++++++++++++++++++++++++++++++++++")
    print("DataProcess : {}".format(config.process))
    print("TrainModel : {}".format(config.train))
    print("TestModel : {}".format(config.test))
    print("t_model : {}".format(config.t_model))
    print("t_data : {}".format(config.t_data))
    print("predict : {}".format(config.predict))
    print("++++++++++++++++++++++++++++++++++++++++++++")

    return config


if __name__ == "__main__":
    print("Process ID is {},Parent Process ID is {}".format(os.getpid(), os.getppid()))
    config = parse_print()
    if config.device != cpu_device:
        print("Use GPU for training......")
        device_number = config.device[-1]
        torch.cuda.set_device(int(device_number))
        print("Currennt Cuda Device {}".format(torch.cuda.current_device()))
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)
        print("torch.cuda.initial_seed", torch.cuda.initial_seed)

    main()
