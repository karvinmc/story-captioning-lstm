import torch
import torch.nn as nn


class CaptionLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, feature_size=256, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size + feature_size, hidden_size, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions)  # [B, T, E]
        features = features.unsqueeze(1).repeat(1, embeddings.size(1), 1)
        lstm_input = torch.cat((features, embeddings), dim=2)
        out, _ = self.lstm(lstm_input)
        out = self.dropout(out)
        return self.linear(out)
