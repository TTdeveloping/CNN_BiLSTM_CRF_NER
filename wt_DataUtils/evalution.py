class Eval:
    def __init__(self):
        self.predict_num = 0
        self.correct_num = 0
        self.gold_num = 0

        self.precision = 0
        self.recall = 0
        self.fscore = 0

    def clear_prf(self):
        self.predict_num = 0
        self.correct_num = 0
        self.gold_num = 0

        self.precision = 0
        self.recall = 0
        self.fscore = 0

    def getFscore(self):
        if self.predict_num == 0:
            self.precision = 0
        else:
            self.precision = (self.correct_num / self.predict_num) * 100

        if self.gold_num == 0:
            self.recall = 0
        else:
            self.recall = (self.correct_num / self.gold_num) * 100

        if self.precision + self.recall == 0:
            self.fscore = 0
        else:
            self.fscore = (2 * (self.precision * self.recall)) / (self.precision + self.recall)
        return self.precision, self.recall, self.fscore

    def acc_rate(self):
        return (self.correct_num / self.gold_num) * 100


class EvalPRF:
    def evalution_PRF(self, predict_sent_labels, gold_sent_labels, eval):
        """
        :param predict_sent_labels: p_label 一句话的所有单词预测标签
        :param gold_sent_labels:   g_label 一句话的所有单词金标标签
        :param eval:
        :return:
        """
        # self.get_entity这个函数主要是得到一句话中个单词的实体，并且判断该实体所在位置，如非单个实体，
        # 记录实体所在位置上的id是连续的id号；单个实体记录的是两个同样的id号
        gold_entity = self.get_entity(gold_sent_labels)
        predict_entity = self.get_entity(predict_sent_labels)
        eval.predict_num += len(predict_entity)
        eval.gold_num += len(gold_entity)
        count = 0
        for p_ent in predict_entity:
            if p_ent in gold_entity:
                count += 1
                eval.correct_num += 1

    def get_entity(self, sent_labels):
        idx = 0
        idy = 0
        endpos = -1
        entity = []
        while (idx < len(sent_labels)):
            if (self.judge_start_label(sent_labels[idx])):
                idy = idx
                endpos = -1
                while (idy < len(sent_labels)):
                    if not self.judge_next_label(sent_labels[idy], sent_labels[idx], idy - idx):
                        endpos = idy - 1
                        break
                    endpos = idy
                    idy += 1
                entity.append(self.Label_entity(sent_labels[idx]) + '[' + str(idx) + ',' + str(endpos) + ']')
                idx = endpos
            idx += 1
        return entity

    def judge_next_label(self, label, startLabel, distance):
        if distance == 0:
            return True
        if len(label) < 3:
            return False
        if distance != 0 and self.judge_start_label(label):
            return False
        if (startLabel[0] == 's' or startLabel[0] == 's') and startLabel[1] == '-':
            return False
        if self.Label_entity(label) != self.Label_entity(startLabel):
            return False
        return True

    def Label_entity(self, label):
        start_label = ['B', 'b', 'M', 'm', 'E', 'e', 'S', 's', 'I', 'i']
        if len(label) > 2 and label[1] == '-':
            if label[0] in start_label:
                return label[2:]
        return label

    def judge_start_label(self, word_label):
        start = ['b', 'B', 's', 'S']
        if len(word_label) < 3:
            return False
        else:
            return (word_label[0] in start) and word_label[1] == '-'
