from transformers import CLIPProcessor, CLIPModel
import torch
import torch.nn as nn

class CLIPTransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size=768, num_heads=12, num_layers=6, max_len=100):
        super(CLIPTransformerModel, self).__init__()
        # Load pretrained CLIP model
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        
        # Freeze CLIP to save compute
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Transformer decoder configuration
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_len, embed_size))
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, images, captions, tgt_mask=None):
        # images: (B, 5, 3, 336, 336)
        # captions: (B, seq_len)
        B, S, C, H, W = images.size()
        
        # Process images with CLIP
        images = images.view(B * S, C, H, W)  # (B*S, 3, 336, 336)
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(images.device)
        image_features = self.clip_model.get_image_features(**inputs)  # (B*S, 768)
        image_features = image_features.view(B, S, self.embed_size)  # (B, 5, 768)
        
        # Aggregate image features (e.g., mean pooling)
        image_features = image_features.mean(dim=1)  # (B, 768)
        
        # Process captions
        embedded = self.embedding(captions) + self.pos_encoder[:, :captions.size(1), :]  # (B, seq_len, embed_size)
        embedded = self.dropout(embedded)
        
        # Create target mask for transformer (causal masking)
        if tgt_mask is None:
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(captions.size(1)).to(captions.device)
        
        # Transformer decoder
        output = self.transformer_decoder(
            tgt=embedded.transpose(0, 1),  # (seq_len, B, embed_size)
            memory=image_features.unsqueeze(0),  # (1, B, embed_size)
            tgt_mask=tgt_mask
        ).transpose(0, 1)  # (B, seq_len, embed_size)
        
        return self.linear(output)  # (B, seq_len, vocab_size)

    def generate(self, images, max_len=100, start_token_idx=0, end_token_idx=1, beam_width=3):
        B, S, C, H, W = images.size()
        device = images.device
        
        # Process images with CLIP
        images = images.view(B * S, C, H, W)
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(device)
        image_features = self.clip_model.get_image_features(**inputs)  # (B*S, 768)
        image_features = image_features.view(B, S, self.embed_size).mean(dim=1)  # (B, 768)
        
        # Initialize beams: [sequence, score, tokens]
        beams = [[[], 0.0, [start_token_idx]] for _ in range(B)]
        generated = []
        
        for b in range(B):
            batch_features = image_features[b:b+1]  # (1, 768)
            batch_beams = beams[b:b+1]
            completed = []
            
            for _ in range(max_len):
                new_beams = []
                for seq, score, tokens in batch_beams:
                    if tokens[-1] == end_token_idx:
                        completed.append([seq, score, tokens])
                        continue
                    
                    # Current token
                    input_tensor = torch.tensor([tokens[-1]], device=device).unsqueeze(0)  # (1, 1)
                    embedded = self.embedding(input_tensor) + self.pos_encoder[:, :1, :]  # (1, 1, embed_size)
                    
                    # Transformer forward pass
                    output = self.transformer_decoder(
                        tgt=embedded.transpose(0, 1),  # (1, 1, embed_size)
                        memory=batch_features.unsqueeze(0),  # (1, 1, embed_size)
                        tgt_mask=torch.ones(1, 1, device=device).bool()
                    ).transpose(0, 1)  # (1, 1, embed_size)
                    
                    output = self.linear(output.squeeze(1))  # (1, vocab_size)
                    probs = torch.softmax(output, dim=1).squeeze(0)  # (vocab_size,)
                    top_probs, top_idx = probs.topk(beam_width)
                    
                    for prob, idx in zip(top_probs.tolist(), top_idx.tolist()):
                        new_seq = seq + [idx]
                        new_score = score + torch.log(torch.tensor(prob)).item()
                        new_tokens = tokens + [idx]
                        new_beams.append([new_seq, new_score, new_tokens])
                
                batch_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
                
                if all(tokens[-1] == end_token_idx for _, _, tokens in batch_beams):
                    completed.extend(batch_beams)
                    break
            
            completed.extend(batch_beams)
            best_seq = max(completed, key=lambda x: x[1])[0]
            generated.append(best_seq)
        
        max_seq_len = max(len(seq) for seq in generated)
        output = torch.zeros(B, max_seq_len, dtype=torch.long, device=device)
        for b, seq in enumerate(generated):
            output[b, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        
        return output