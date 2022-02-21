from transformers import AutoTokenizer, AutoModel
import json
import torch
import urllib.request
import json
from timeit import default_timer as timer
from torch.nn.utils.rnn import pad_sequence
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SRC_SEQ_LEN = 256
TGT_SEQ_LEN = 128


def get_vocab(url):
    resp = urllib.request.urlopen(url)
    ele_json = json.loads(resp.read())
    return ele_json


def clean_word(word):
    temp = str()
    for char in word:
        if char.isalpha():
            temp = temp+char
    return temp.lower()


def json2corpus(path, num_data):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    file = open(path)
    src_corpus = []
    tgt_corpus = []
    i = 0
    for line in file.readlines():
        if i < num_data:
            pl_tokens = tokenizer.tokenize(json.loads(line)['code'])
            nl_tokens = json.loads(line)['docstring']
            words = nl_tokens.split(' ')
            words = [clean_word(word.replace('\n', '')) for word in words if word != '']
            src_corpus.append(pl_tokens)
            tgt_corpus.append(words)
            i += 1
        else:
            break
    return src_corpus, tgt_corpus


def tokens2id(data, vocab):
    return [vocab[word] for word in data]


def sen2ids(sentence, vocab, tokenizer):
    token = [vocab['<s>']]
    token.extend(tokens2id(tokenizer.tokenize(sentence), vocab))
    token.append(vocab['</s>'])
    return token


def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src, tgt in batch:
        src_batch.append(src[: SRC_SEQ_LEN])
        tgt_batch.append(tgt[: TGT_SEQ_LEN])
    src_batch = pad_sequence(src_batch)
    tgt_batch = pad_sequence(tgt_batch)
    return src_batch, tgt_batch


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(pl_tokens_ids, nl_tokens_ids):
    src_seq_len = pl_tokens_ids.shape[0]
    tgt_seq_len = nl_tokens_ids.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (pl_tokens_ids == torch.tensor(0)).transpose(0, 1)
    tgt_padding_mask = (nl_tokens_ids == torch.tensor(0)).transpose(0, 1)

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def train_epoch(model, optimizer, train_dataloader, loss_fn):
    model.train()
    losses = 0
    for src, tgt in train_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt)
        logits = model(src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(train_dataloader)


def evaluate(model, val_dataloader, loss_fn):
    model.eval()
    losses = 0

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt)

        logits = model(src, tgt, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
        losses += loss.item()

    return losses / len(val_dataloader)


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)

    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)

        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == 2:
            break
    return ys


def translate(model, src_sentence, src_vocab, tgt_vocab, tokenizer):
    model.eval()
    src = torch.tensor(sen2ids(src_sentence, src_vocab, tokenizer)).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens, device=DEVICE)).type(torch.bool)

    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=1).flatten()
    return tgt_tokens


def generate_eval_sentence(path, len_data):
    path = path
    file = open(path)
    len = 0
    src_sentences = []
    tgt_sentences = []
    for line in file.readlines():
        if len < len_data:
            src_sentence = json.loads(line)['code']
            tgt_sentence = json.loads(line)['docstring']
            src_sentences.append(src_sentence)
            tgt_sentences.append(tgt_sentence)
            len += 1
        else:
            break
    return src_sentences, tgt_sentences
