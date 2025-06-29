import torch
import torch.nn as nn


class LSTMDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(LSTMDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.feat_proj = nn.Linear(embed_size * 5, embed_size)  # <-- Added
        self.lstm = nn.LSTM(embed_size * 2, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        # features: (B, 5, embed_size)
        # captions: (B, seq_len)
        B, S, _ = features.shape
        combined = features.view(B, -1)  # (B, 5 * embed_size)
        feat_embed = self.feat_proj(combined).unsqueeze(1)  # (B, 1, embed_size)

        embedded = self.embedding(captions)  # (B, seq_len, embed_size)
        repeated_feat = feat_embed.expand(-1, embedded.size(1), -1)
        lstm_input = torch.cat([embedded, repeated_feat], dim=2)
        out, _ = self.lstm(lstm_input)
        return self.linear(out)  # (B, seq_len, vocab_size)
