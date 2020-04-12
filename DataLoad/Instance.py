import torch
import random
from wt_DataUtils.UniversalData import *

torch.manual_seed(seed_num)
random.seed(seed_num)


class Instance():
    def __init__(self):
        self.words = []
        self.chars = []
        self.labels = []
        self.words_size = 0
        self.chars_size = 0
        self.words_index = []
        self.chars_index = []
        self.labels_index = []