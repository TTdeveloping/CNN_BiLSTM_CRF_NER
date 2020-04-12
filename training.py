from wt_DataUtils.Optim import *
import torch.nn as nn
import time
import torch.nn.utils as utils
from wt_DataUtils.evalution import *
from wt_DataUtils.utils import *
import random
import sys


class Train(object):
    """
        Train
    """
    def __init__(self, **kwargs):  # kwargs 就是将传递进来的一堆参数按照字典的形式存储起来。
        """
        :param kwargs:
        """
        print("Train is starting..........")
        self.train_iter = kwargs["train_iter"]
        self.dev_iter = kwargs["dev_iter"]
        self.test_iter = kwargs["test_iter"]
        self.model = kwargs["model"]
        self.config = kwargs["config"]
        self.use_crf = self.config.use_crf
        self.average_batch = self.config.average_batch
        self.early_max_stopping = self.config.early_stopping
        self.optimizer = Optimizer(name=self.config.learning_algorithm, model=self.model, lr=self.config.learning_rate,
                                   weight_decay=self.config.weight_decay, grad_clip=self.config.clip_max_norm)

        self.loss_function = self._loss(learning_algorithm=self.config.learning_algorithm,
                                        label_paddingId=self.config.label_paddingId, use_crf=self.use_crf)
        print(self.optimizer)
        print(self.loss_function)
        self.best_score = Best_Result()
        self.train_eval, self.dev_eval, self.test_eval = Eval(), Eval(), Eval()
        self.train_iter_len = len(self.train_iter)

    def _loss(self, learning_algorithm, label_paddingId, use_crf=False):
        """
        :param learning_algorithm:
        :param label_paddingId:
        :param use_crf:
        :return:
        """
        if use_crf:
            loss_function = self.model.crf_layer.neg_log_likelihood_loss
            return loss_function
        elif learning_algorithm == "SGD":
            loss_function = nn.CrossEntropyLoss(ignore_index=label_paddingId, reduction="sum")
            return loss_function
        else:
            loss_function = nn.CrossEntropyLoss(ignore_index=label_paddingId, reduction="mean")
            return loss_function

    def train(self):
        """
        :return:

        """
        epochs = self.config.epochs
        clip_max_norm_use = self.config.clip_max_norm_use
        clip_max_norm = self.config.clip_max_norm
        new_lr = self.config.learning_rate

        for epoch in range(1, epochs + 1):
            print("\n## The {} Epoch, All {} Epochs ! ##".format(epoch, epochs))
            self.optimizer = self._decay_learning_rate(epoch=epoch - 1, init_lr=self.config.learning_rate)
            print("now learning rate is {}\n".format(self.optimizer.param_groups[0].get("lr")), end="")
            start_time = time.time()
            random.shuffle(self.train_iter)
            self.model.train()
            steps = 1
            backward_count = 0
            for batch_count, batch_features in enumerate(self.train_iter):
                backward_count += 1
                word, char, mask, sentence_length, tags = self._get_model_args(batch_features)
                logit = self.model(word, char, sentence_length, train=True)
                loss = self._calculate_loss(logit, mask, tags)
                loss.backward()
                self._clip_model_norm(clip_max_norm_use, clip_max_norm)
                self._optimizer_batch_step(config=self.config, backward_count=backward_count)
                steps += 1
                if (steps - 1) % self.config.log_interval == 0:
                    self.getAcc(self.train_eval, batch_features, logit, self.config)
                    sys.stdout.write("\nbatch_count = [{}] , loss is {:.6f}, [TAG-ACC is {:.6f}%]"
                                     .format(backward_count + 1, loss.item(), self.train_eval.acc_rate()))
            end_time = time.time()
            print("\nTrain Time {: .3f}".format(end_time - start_time), end="")
            print()
            self.evalution(model=self.model, epoch=epoch, config=self.config)
            self._model2file(model=self.model, epoch=epoch, config=self.config)
            self._early_stopping(epoch=epoch)

    def _early_stopping(self, epoch):
        best_epoch = self.best_score.best_epoch
        if epoch > best_epoch:
            self.best_score.early_stopping += 1
            print("How many rounds  has the performance of the dev set not gone up? {} /{} ".format
                  (self.best_score.early_stopping, self.early_max_stopping))
            if self.best_score.early_stopping >= self.early_max_stopping:
                print("Early Stop Train . Best Score Local on {} Epoch.".format(self.best_score.best_epoch))
                exit()

    def _model2file(self, model, config, epoch):
        """
        :param model:
        :param config:
        :param epoch:
        :return:
        """
        if config.save_model and config.save_all_model:
            save_model_all(model, config.save_dir, config.model_name, epoch)
        elif config.save_model and config.save_best_model:
            save_best_model(model, config.save_best_model_path, config.model_name, self.best_score)
        else:
            print()

    def evalution(self, model, epoch, config):
        """
        :param model:
        :param epoch:
        :param config:
        :return:
        """
        self.dev_eval.clear_prf()
        eval_start_time = time.time()
        self.eval_batch(self.dev_iter, model, self.dev_eval, self.best_score, epoch, config, test=False)
        eval_end_time = time.time()
        print("Dev Time {:.3f}".format(eval_end_time - eval_start_time))

        self.test_eval.clear_prf()
        eval_start_time = time.time()
        self.eval_batch(self.test_iter, model, self.test_eval, self.best_score, epoch, config, test=True)
        eval_end_time = time.time()
        print("Test Time {:.3f}".format(eval_end_time - eval_start_time))

    def eval_batch(self, data_iter, model, eval_instance, best_score, epoch, config, test=False):
        """
        :param data_iter:
        :param model:
        :param eval_instance:
        :param best_score:
        :param epoch:
        :param config:
        :param test:
        :return:
        """
        model.eval()  # 让模型知道何时切换到eval模式
        eval_acc = Eval()
        eval_PRF = EvalPRF()
        gold_labels = []
        predict_labels = []
        for batch_features in data_iter:
            word, char, mask, sentence_length, tags = self._get_model_args(batch_features)
            logit = model(word, char, sentence_length, train=False)
            if self.use_crf is False:

                # list[[], [], ...[]]    len(list)=batch_size,
                # 每个小列表中存放单词被预测的标签id
                predict_indices = predict_tag_id(logit)

                for id_inst in range(batch_features.batch_length):
                    inst = batch_features.batch[id_inst]
                    label_indices = predict_indices[id_inst]  # [取出每一句话的单词对应的id]
                    predict_label = []
                    for id_word in range(inst.words_size):
                        # 存放每句话中所有单词的预测标签
                        predict_label.append(config.create_alphabet.label_alphabet.from_id(label_indices[id_word]))
                    gold_labels.append(inst.labels)  # 存放每句话中单词的金标标签 [[], [],...[]]
                    predict_labels.append(predict_label)  # 存放每句话中单词的预测标签[[], [],...[]]

            else:
                pass  # 写 crf 的时候补充
        for p_label, g_label in zip(predict_labels, gold_labels):  # 一句话一句话的金标和预测标签拿出来
            eval_PRF.evalution_PRF(predict_sent_labels=p_label, gold_sent_labels=g_label, eval=eval_instance)
        if eval_acc.gold_num == 0:
            eval_acc.gold_num = 1
        p, r, f = eval_instance.getFscore()
        test_flag = "Test"
        if test is False:
            print()
            This_Test = "Dev"
            best_score.current_dev_score = f
            if f >= best_score.best_dev_score:
                best_score.best_dev_score = f
                best_score.best_epoch = epoch
                best_score.best_test = True
            if test is True and best_score.best_test is True:
                best_score.p = p
                best_score.r = r
                best_score.f = f
            print(
                " {} evalution: precision = {:.6f}% recall = {:.6f}%, f-score = {:.6f}%,   [TAG-ACC = {:.6f}]".format
                (This_Test, p, r, f, 0.0000)
            )
        if test is True:
            print("Current Best Dev F-score: {:.6f}, Locate on {} Epoch.".format(best_score.best_dev_score, best_score.best_epoch))
            print("The Current Best Test Result: precision = {:.6f}%, f-score = {:.6f}%".format(
                best_score.p, best_score.r, best_score.f))
        if test is True:
            best_score.best_test = False

    def _get_model_args(self, batch_features):
        """
        :param batch_features:
        :return:
        """
        word = batch_features.word_features
        char = batch_features.char_features
        mask = word > 0
        sentence_length = batch_features.Origin_sentence_length
        tags = batch_features.label_features
        return word, char, mask, sentence_length, tags

    def _decay_learning_rate(self, epoch, init_lr):
        """
        :param epoch:
        :param init_lr:
        :return:
        """
        lr = init_lr / (1 + self.config.lr_rate_decay * epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return self.optimizer

    def _calculate_loss(self, prediction, mask, tags):
        """
        :param prediction:
        :param mask:
        :param tags:
        :return:
        """
        if not self.use_crf:
            batch_size, max_sentence_len = prediction.size(0), prediction.size(1)
            lstm_prediction = prediction.view(batch_size * max_sentence_len, -1)  # [640, 18]
            change_tags = tags.view(-1)

            return self.loss_function(lstm_prediction, change_tags)

    def _clip_model_norm(self, clip_max_norm_use, clip_max_norm):
        if clip_max_norm_use is True:
            gclip = None if clip_max_norm == 'None' else float(clip_max_norm)
            assert isinstance(gclip, float)
            utils.clip_grad_norm_(self.model.parameters(), max_norm=gclip)

    def _optimizer_batch_step(self, config, backward_count):
        """
        :param config:
        :param backward_count:
        :return:
        """
        if backward_count % config.backward_batch_size == 0 or backward_count == self.train_iter_len:
            self.optimizer.step()
            self.optimizer.zero_grad()

    @staticmethod
    def getAcc(eval_acc, batch_features, logit, config):
        """
        :param eval_acc:
        :param batch_features:
        :param logit:
        :param config:
        :return:
        """
        eval_acc.clear_prf()
        predict_ids = predict_tag_id(logit)  # predict_ids [16,40]存放每个单词预测的最大值的id
        for id_inst in range(batch_features.batch_length):
            inst = batch_features.batch[id_inst]
            label_ids = predict_ids[id_inst]  # label_ids is 存放40个单词预测标签id的列表
            predict_label = []
            gold_label = inst.labels
            for id_word in range(inst.words_size):
                predict_label.append(config.create_alphabet.label_alphabet.from_id(label_ids[id_word]))
                # accroding id find label and add them into predict_label list
            assert len(predict_label) == len(gold_label)
            correct = 0
            for p_label, g_label in zip(predict_label, gold_label):
                if p_label == g_label:
                    correct +=1
            eval_acc.correct_num += correct
            eval_acc.gold_num += len(gold_label)











