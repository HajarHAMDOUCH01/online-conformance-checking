import torch
import torch.nn as nn
# this code is not currently used 
PAD, SOS, EOS, UNK = "<PAD>", "<SOS>", "<EOS>", "<UNK>"


class Vocab:
    def __init__(self):
        self.t2i = {PAD: 0, SOS: 1, EOS: 2, UNK: 3}
        self.i2t = {v: k for k, v in self.t2i.items()}

    def add(self, tok):
        if tok not in self.t2i:
            i = len(self.t2i)
            self.t2i[tok] = i
            self.i2t[i] = tok

    def encode(self, toks):
        return [self.t2i.get(t, self.t2i[UNK]) for t in toks]

    def decode(self, ids):
        return [self.i2t.get(i, UNK) for i in ids]

    def __len__(self):
        return len(self.t2i)