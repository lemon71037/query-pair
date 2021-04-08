import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class QueryPairModel(nn.Module):
    def __init__(self):
        super(QueryPairModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=600, nhead=6, dim_feedforward=2048)
        self.encoder = nn.Sequential(
            nn.Embedding(22000, 600),  # (seq_len, bs, 300)
            PositionalEncoding(600, max_len=100),
            nn.TransformerEncoder(self.encoder_layer, 6)  # seq_len, bs, 1024
        )
        self.decoder = nn.Sequential(
            nn.Linear(1200, 512, bias=True),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1, bias=False)
        )

    def forward(self, q1, q2):
        q1 = q1.transpose(1, 0)  # seq_len, bs
        q2 = q2.transpose(1, 0)
        encoded_q1 = self.encoder(q1)[-1]
        encoded_q2 = self.encoder(q2)[-1]

        cat_q = torch.cat((encoded_q1, encoded_q2), dim=1)  # bs, embed_size * 2
        decoded_q = self.decoder(cat_q)
        return decoded_q


if __name__ == "__main__":
    q1 = torch.randint(low=1, high=10000, size=(2, 70)).cuda()  # bs, seq_len
    q2 = torch.randint(low=1, high=10000, size=(2, 50)).cuda()
    model = QueryPairModel().cuda()
    model(q1, q2)
