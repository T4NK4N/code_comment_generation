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
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    train_path = 'dataset/python/train.jsonl'
    val_path = 'dataset/python/valid.jsonl'
    num_data = 50000
    src_corpus, tgt_corpus = json2corpus(train_path, num_data)
    vocab = Vocab()
    src_vocab = vocab.from_corpus(src_corpus, 2)
    tgt_vocab = vocab.from_corpus(tgt_corpus, 2)
    

    torch.manual_seed(0)
    EMBED_SIZE = 512
    NHEAD = 8
    NUM_ENCODER_LAYERS = 6
    NUM_DECODER_LAYERS = 6
    SRC_VOCAB_SIZE = len(src_vocab)
    TGT_VOCAB_SIZE = len(tgt_vocab)

    # dataset
    len_train_data = 50000
    len_val_data = 10000
    train_dataset = code2comments(train_path, len_train_data, src_vocab, tgt_vocab)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    del (train_dataset)
    val_dataset = code2comments(val_path, len_val_data, src_vocab, tgt_vocab)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    del (val_dataset)

    # transformer
    transformer = code2commentTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMBED_SIZE, NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        print(p.size())
    transformer = transformer.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-9)
    best_loss = 10
    # train
    NUM_EPOCHS = 15
    len_data = 10
    src_sentences, tgt_sentences = generate_eval_sentence(train_path, len_data)
    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer, train_dataloader, loss_fn)
        end_time = timer()
        val_loss = evaluate(transformer, val_dataloader, loss_fn)
        print((
                  f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
        for src in src_sentences:
            tgt_tokens = translate(transformer, src_sentences[0], src_vocab, tgt_vocab, tokenizer)
            print(tgt_tokens)
        if val_loss < best_loss:
            torch.save(transformer, 'transformer.pt')
            best_loss = val_loss































