import torch
import torch.nn as nn

# In models/decoder.py
class LSTMDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(LSTMDecoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = nn.Linear(embed_size + hidden_size, hidden_size)
        self.attention_score = nn.Linear(hidden_size, 1)
        self.attention_dropout = nn.Dropout(0.3)  # Add dropout
        self.lstm = nn.LSTM(embed_size * 2, hidden_size, num_layers, batch_first=True, dropout=0.3 if num_layers > 1 else 0)
        self.dropout = nn.Dropout(0.3)  # Add dropout before linear
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions, hidden=None):
        B, seq_len = captions.size()
        embedded = self.embedding(captions)
        outputs = []
        if hidden is None:
            hidden = (torch.zeros(self.num_layers, B, self.hidden_size).to(features.device),
                      torch.zeros(self.num_layers, B, self.hidden_size).to(features.device))

        for t in range(seq_len):
            current_embed = embedded[:, t, :].unsqueeze(1)
            h = hidden[0][-1].unsqueeze(1).expand(-1, features.size(1), -1)
            attn_input = torch.cat([features, h], dim=2)
            attn_hidden = torch.tanh(self.attention(attn_input))
            attn_scores = self.attention_score(attn_hidden).squeeze(2)
            attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(2)
            attn_weights = self.attention_dropout(attn_weights)  # Apply dropout
            context = (attn_weights * features).sum(dim=1).unsqueeze(1)
            lstm_input = torch.cat([current_embed, context], dim=2)
            out, hidden = self.lstm(lstm_input, hidden)
            outputs.append(out)

        outputs = torch.cat(outputs, dim=1)
        outputs = self.dropout(outputs)  # Apply dropout
        return self.linear(outputs)

    def generate(self, features, max_len=100, start_token_idx=0, end_token_idx=1, beam_width=3):
        # Generate captions using beam search
        # features: (B, 5, embed_size) - batch of image features
        # max_len: maximum length of generated sequence
        # start_token_idx: index of <SOS> token
        # end_token_idx: index of <EOS> token
        # beam_width: number of sequences to track
        B = features.size(0)
        device = features.device

        # Initialize beam: list of [sequence, score, hidden_state, inputs]
        beams = [[[], 0.0, (
            torch.zeros(self.num_layers, 1, self.hidden_size).to(device),
            torch.zeros(self.num_layers, 1, self.hidden_size).to(device)
        ), [start_token_idx]] for _ in range(B)]

        # Process each batch element independently
        generated = []
        for b in range(B):
            batch_features = features[b:b+1]  # (1, 5, embed_size)
            batch_beams = beams[b:b+1]  # Initialize with single beam
            completed = []

            for _ in range(max_len):
                new_beams = []
                for seq, score, hidden, inputs in batch_beams:
                    # If sequence ended with <EOS>, add to completed
                    if inputs[-1] == end_token_idx:
                        completed.append([seq, score, hidden, inputs])
                        continue

                    # Current input
                    input_tensor = torch.tensor([inputs[-1]]).unsqueeze(0).to(device)  # (1, 1)
                    embedded = self.embedding(input_tensor)  # (1, 1, embed_size)

                    # Compute attention weights
                    h = hidden[0][-1].unsqueeze(1).expand(-1, batch_features.size(1), -1)  # (1, 5, hidden_size)
                    attn_input = torch.cat([batch_features, h], dim=2)  # (1, 5, embed_size + hidden_size)
                    attn_hidden = torch.tanh(self.attention(attn_input))  # (1, 5, hidden_size)
                    attn_scores = self.attention_score(attn_hidden).squeeze(2)  # (1, 5)
                    attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(2)  # (1, 5, 1)
                    context = (attn_weights * batch_features).sum(dim=1).unsqueeze(1)  # (1, 1, embed_size)

                    # Concatenate input with context
                    lstm_input = torch.cat([embedded, context], dim=2)  # (1, 1, embed_size * 2)

                    # Feed into LSTM
                    out, new_hidden = self.lstm(lstm_input, hidden)  # out: (1, 1, hidden_size)

                    # Project to vocabulary
                    out = self.linear(out.squeeze(1))  # (1, vocab_size)
                    probs = torch.softmax(out, dim=1).squeeze(0)  # (vocab_size,)
                    top_probs, top_idx = probs.topk(beam_width)  # Top k probabilities and indices

                    # Add new candidates to beams
                    for prob, idx in zip(top_probs.tolist(), top_idx.tolist()):
                        new_seq = seq + [idx]
                        new_score = score + torch.log(torch.tensor(prob)).item()
                        new_inputs = inputs + [idx]
                        new_beams.append([new_seq, new_score, new_hidden, new_inputs])

                # Select top beam_width beams
                batch_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

                # If all beams have ended, stop
                if all(inputs[-1] == end_token_idx for _, _, _, inputs in batch_beams):
                    completed.extend(batch_beams)
                    break

            # Add remaining beams to completed
            completed.extend(batch_beams)

            # Select best sequence
            best_seq = max(completed, key=lambda x: x[1])[0]
            generated.append(best_seq)

        # Convert to tensor
        max_seq_len = max(len(seq) for seq in generated)
        output = torch.zeros(B, max_seq_len, dtype=torch.long).to(device)
        for b, seq in enumerate(generated):
            output[b, :len(seq)] = torch.tensor(seq, dtype=torch.long)

        return output