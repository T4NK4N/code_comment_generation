import torch
import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
from utils import *


class code2comments(Dataset):
    def __init__(self, path, len_data, src_vocab, tgt_vocab):
        self.path = path
        self.file = open(path)
        self.len_data = len_data
        self.nl_tokens = []
        self.pl_tokens = []
        len = 0
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        for line in self.file.readlines():
            if len < self.len_data:
                src_sentence = json.loads(line)['code']
                tgt_sentence = json.loads(line)['docstring']
                src_ids = sen2ids(src_sentence, src_vocab, tokenizer)
                tgt_ids = sen2ids(tgt_sentence, tgt_vocab, tokenizer)
                self.pl_tokens.append(src_ids)
                self.nl_tokens.append(tgt_ids)
                len += 1
            else:
                break
        del(tokenizer)

    def __len__(self):
        return len(self.nl_tokens)

    def __getitem__(self, index):
        nl_tokens_ids = self.nl_tokens[index]
        pl_tokens_ids = self.pl_tokens[index]
        return torch.tensor(pl_tokens_ids), torch.tensor(nl_tokens_ids)





