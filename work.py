from dataset import code2comments
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
from transformer import *
import urllib.request
import json
from utils import *
import torch.nn as nn
from vocab import Vocab


def build_corpus(path):
    file = open(path)
    coprus = dict()
    for line in file.readlines():
        src_sentence = json.loads(line)['code']
        tgt_sentence = json.loads(line)['docstring']
        coprus[tgt_sentence] = src_sentence
    return coprus


if __name__ == '__main__':
    path1 = 'dataset/python/test.jsonl'
    corpus = build_corpus(path1)
    path2 = 'models/codebert/test_1.gold'
    file = open(path2)
    for line in file.readlines():
        number = line.split('\t')[0]
        lines = line.split('\t')[1].split('\n')[0]
        lines = lines.replace(' .', '.')
        try:
            print(number)
            print(corpus[lines])
        except KeyError as error:
            pass
        print('-------------------------------------')


    # for comment in comments:
    #     try:
    #         print(corpus[comment])
    #     except KeyError as error:
    #         print(error)
    #     print('-------------------------------------')
    # pass