from collections import Counter
from itertools import chain
import json
from transformers import AutoTokenizer, AutoModel
from utils import *
from dataset import code2comments
from torch.utils.data import DataLoader, Dataset


class Vocab(object):
    def __init__(self):
        self.token2id = dict()
        self.token2id['<pad>'] = 0  # Pad Token
        self.token2id['<s>'] = 1  # Start Token
        self.token2id['</s>'] = 2  # End Token
        self.token2id['<unk>'] = 3  # Unknown Token

        self.id2token = {v: k for k, v in self.token2id.items()}

    def __len__(self):
        """ Compute number of words in VocabEntry.
        @returns len (int): number of words in VocabEntry
        """
        return len(self.token2id)

    def __getitem__(self, word):
        """ Retrieve word's index. Return the index for the unk
        token if the word is out of vocabulary.
        @param word (str): word to look up.
        @returns index (int): index of word
        """
        return self.token2id.get(word, 3)

    def __contains__(self, word):
        """ Check if word is captured by VocabEntry.
        @param word (str): word to look up
        @returns contains (bool): whether word is contained
        """
        return word in self.token2id

    def add(self, word):
        """ Add word to VocabEntry, if it is previously unseen.
        @param word (str): word to add to VocabEntry
        @return index (int): index that the word has been assigned
        """
        if word not in self:
            wid = self.token2id[word] = len(self)
            self.id2token[wid] = word
            return wid
        else:
            return self[word]

    @staticmethod
    def from_corpus(corpus, freq_cutoff):
        """ Given a corpus construct a Vocab Entry.
        @param corpus (list[str]): corpus of text produced by read_corpus function
        @param size (int): # of words in vocabulary
        @param freq_cutoff (int): if word occurs n < freq_cutoff times, drop the word
        @returns vocab_entry (VocabEntry): VocabEntry instance produced from provided corpus
        """
        vocab = Vocab()
        word_freq = Counter(chain(*corpus))

        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        print('number of word types: {}, number of word types w/ frequency >= {}: {}'
              .format(len(word_freq), freq_cutoff, len(valid_words)))
        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)
        for word in top_k_words:
            vocab.add(word)
        return vocab


if __name__ == '__main__':
    path = 'dataset/python/train.jsonl'
    num_data = 50000

    src_corpus, tgt_corpus = json2corpus(path, num_data)
    vocab = Vocab()
    src_vocab = vocab.from_corpus(src_corpus, 1)
    tgt_vocab = vocab.from_corpus(tgt_corpus, 1)
    # dataset = code2comments(path, 3, vocab)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    # for src, tgt in dataloader:
    #     print(src.size())
    #     print(tgt.size())



