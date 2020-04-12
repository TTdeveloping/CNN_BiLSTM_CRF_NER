import torch
import os


class Best_Result:
    """
    Best_Result
    """
    def __init__(self):
        self.current_dev_score = -1
        self.best_dev_score = -1
        self.best_score = -1
        self.best_epoch = -1
        self.best_test = False
        self.early_stopping = 0
        self.p = -1
        self.r = -1
        self.f = -1


def predict_tag_id(output):
    """
    :param output:  logit
    :return:
    """
    print("LOOK AT ....")

    batch_size = output.size(0)
    _, max_indices = torch.max(output, dim=2)
    label = []
    for i in range(batch_size):  # batch_size是一个批的大小。
        label.append(max_indices[i].cpu().data.numpy())
    return label


def save_model_all(model, save_dir, model_name, epoch):
    """
    :param model:
    :param save_dir:
    :param model_name:
    :param epoch:
    :return:
    """
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, model_name)
    save_path = '{}_epoch_{}.pt'.format(save_prefix, epoch)
    print("save all model to {}".format(save_path))
    output = open(save_path, mode="wb")
    torch.save(model.state_dict(), output)
    output.close()


def save_best_model(model, save_dir, model_name, best_eval):
    """
    :param model:
    :param save_dir:
    :param model_name:
    :param best_evaltion:
    :return:
    """
    if best_eval.current_dev_score >= best_eval.best_dev_score:
        if not os.path.isdir(save_dir): os.makedirs(save_dir)
        model_name = "{}.pt".format(model_name)
        save_path = os.path.join(save_dir, model_name)
        print("Save best model to {}".format(save_path))
        output = open(save_path, mode="wb")
        torch.save(model.state_dict(), output)  # 官方推荐方法，只保存模型中的参数。
        output.close()
        best_eval.early_stopping = 0