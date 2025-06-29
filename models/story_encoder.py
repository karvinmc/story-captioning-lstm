import torch.nn as nn

class StoryEncoder(nn.Module):
    def __init__(self, feature_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_dim, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
    def forward(self, features):
        # features: [batch, seq_len, feature_dim]
        _, (h_n, _) = self.lstm(features)
        out = self.dropout(h_n[-1])
        return out  # [batch, hidden_dim]
