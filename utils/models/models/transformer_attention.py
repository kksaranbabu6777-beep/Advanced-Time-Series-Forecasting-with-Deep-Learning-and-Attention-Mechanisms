import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    def _init_(self, d_model, max_len=5000):
        super()._init_()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TimeSeriesTransformer(nn.Module):
    def _init_(self, input_dim, d_model=64, nhead=4, num_layers=2, output_dim=1):
        super()._init_()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, output_dim)
    def forward(self, src):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        x = self.transformer(src)
        out = self.fc(x[:, -1, :])
        return out
