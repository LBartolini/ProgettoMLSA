import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model, DEVICE='cpu'):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model).to(DEVICE)
        position = torch.arange(0, max_len).float().unsqueeze(1).to(DEVICE)
        slope = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)).to(DEVICE)
        pe[:, 0::2] = torch.sin(position * slope) # even dimensions
        pe[:, 1::2] = torch.cos(position * slope) # odd dimensions
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x is N, L, D
        # pe is 1, maxlen, D
        scaled_x = x * np.sqrt(self.d_model)
        encoded = scaled_x + self.pe[:, :x.size(1), :]
        return encoded
    
class TransformerModel(nn.Module):
    def __init__(self, transformer, input_len, target_len, dict_size, embedding_dim=150, pretrained_emb=None, DEVICE='cpu'):
        super().__init__()
        self.input_len = input_len
        self.target_len = target_len

        if not pretrained_emb:
            self.emb = nn.Embedding(dict_size, embedding_dim).to(DEVICE)
        else:
            self.emb = nn.Embedding.from_pretrained(pretrained_emb.weight, freeze=False).to(DEVICE)

        self.proj = nn.Linear(embedding_dim, transformer.d_model).to(DEVICE)

        max_len = max(input_len, target_len)
        self.pe = PositionalEncoding(max_len, transformer.d_model, DEVICE)

        self.norm = nn.LayerNorm(transformer.d_model).to(DEVICE)

        self.transf = transformer
        #self.trg_masks = transformer.generate_square_subsequent_mask(target_len)

        self.linear = nn.Linear(transformer.d_model, dict_size).to(DEVICE)

    def preprocess(self, seq):
        emb = self.emb(seq)
        seq_proj = self.proj(emb)
        seq_enc = self.pe(seq_proj)
        return self.norm(seq_enc)

    def encode_decode(self, source, target, source_mask=None, target_mask=None):
        # Projections
        src = self.preprocess(source)
        tgt = self.preprocess(target)

        # Transformer
        out = self.transf(src, tgt,
                          src_key_padding_mask=source_mask,
                          tgt_mask=target_mask)

        # Linear
        relu = nn.ReLU()
        out = self.linear(relu(out)).squeeze(1) # N, L, F
        #softmax = nn.Softmax(dim=1)
        return out

    def predict(self, source_seq, source_mask=None):
        inputs = source_seq[:, -1:]
        for i in range(self.target_len):
            out = self.encode_decode(source_seq, inputs,
                                     source_mask=source_mask,
                                     target_mask=None) # self.trg_masks[:i+1, :i+1]
            inputs = out.detach()
        return nn.Softmax(dim=1)(out)

    def forward(self, X, source_mask=None):
        #self.trg_masks = self.trg_masks.type_as(X)
        source_mask = source_mask.type_as(X) if source_mask is not None else None

        if self.training:
            shifted_target_seq = X[:, self.input_len-1:]
            outputs = self.encode_decode(X, shifted_target_seq,
                                         source_mask=source_mask,
                                         target_mask=None) # self.trg_masks
        else:
            outputs = self.predict(X, source_mask)

        return outputs
