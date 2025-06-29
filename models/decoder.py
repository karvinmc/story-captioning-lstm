# models/decoder.py
import torch
import torch.nn as nn


class LSTMDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(LSTMDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            embed_size + embed_size, hidden_size, num_layers, batch_first=True
        )
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        # captions: list of 5 caption tensors
        B, S, _ = features.shape
        outputs = []
        for i in range(S):
            cap = captions[i]
            embedded = self.embedding(cap[:, :-1])  # remove EOS
            feats = features[:, i].unsqueeze(1).expand(-1, embedded.size(1), -1)
            lstm_input = torch.cat([embedded, feats], dim=2)
            out, _ = self.lstm(lstm_input)
            out = self.linear(out)
            outputs.append(out)
        return outputs
