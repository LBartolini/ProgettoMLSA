import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, dictionary_size, embedding_dim=512, encoding_dim=512, lstm_layer_size=512, lstm_num_layers=2, linear_layer_size=1024, batch_size=16, batch_first=True, bidirectional=True, pretrained_emb=None, DEVICE='cpu'):
        super().__init__()

        if not pretrained_emb:
            self.emb = nn.Embedding(dictionary_size, embedding_dim).to(DEVICE)
        else:
            self.emb = nn.Embedding.from_pretrained(pretrained_emb.weight, freeze=False).to(DEVICE)

        self.encoder = nn.Linear(embedding_dim, encoding_dim).to(DEVICE)
        self.dropout1 = nn.Dropout(p=0.2)

        self.lstm = nn.LSTM(encoding_dim, lstm_layer_size, num_layers=lstm_num_layers, batch_first=batch_first, bidirectional=bidirectional).to(DEVICE)

        self.linear1 = nn.Linear(2*lstm_layer_size if bidirectional else lstm_layer_size, linear_layer_size).to(DEVICE)
        self.dropout2 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(linear_layer_size, dictionary_size).to(DEVICE)

    def forward(self, input_seq):
        relu = nn.ReLU()

        emb = self.emb(input_seq)
        encoded = self.dropout1(relu(self.encoder(emb)))
        lstm_out, _ = self.lstm(encoded)

        #softmax = nn.Softmax(dim=1)

        last = self.linear1(relu(lstm_out[:, -1]))
        predictions = self.linear2(self.dropout2(relu(last)))

        #predictions = softmax(last)

        return predictions

