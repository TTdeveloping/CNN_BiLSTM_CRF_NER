import torch
import random
from wt_DataUtils.UniversalData import seed_num
from wt_DataUtils.UniversalData import cpu_device
torch.manual_seed(seed_num)
random.seed(seed_num)


def prepare_pack_padded_sequence(input_words, sentence_length, device='cpu', descending=True):
    """
    :param input_embed_embedding:
    :param sentence_length:
    :param device:
    :param descending: 降序
    :return:
    """
    order_senten_lengths, indices = torch.sort(torch.Tensor(sentence_length).long(), descending=descending)
    # 降序的句子长度，和句子对应下标
    if device != cpu_device:
        order_senten_lengths, indices = order_senten_lengths.cuda(), indices.cuda()
    _, disorder_indices = torch.sort(indices, descending=False)  #
    order_input_words = input_words[indices]
    return order_input_words, order_senten_lengths.cpu().numpy(), disorder_indices
