import torch.nn as nn

class StoryEncoder(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_dim, batch_first=True)
    def forward(self, features):
        # features: [batch, seq_len, feature_dim]
        _, (h_n, _) = self.lstm(features)
        return h_n[-1]  # [batch, hidden_dim]
