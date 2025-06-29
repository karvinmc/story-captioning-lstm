import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from preprocessing.story_dataset import StoryDataset
from models.encoder import CNNEncoder
from models.decoder import LSTMDecoder
from utils.vocab import Vocabulary
from utils.plot import plot_losses
from utils.checkpoint import save_checkpoint, load_checkpoint
from tqdm import tqdm
import os
import json

# --- Config ---
EPOCHS = 5
BATCH_SIZE = 8
EMBED_SIZE = 256
HIDDEN_SIZE = 512
FREQ_THRESHOLD = 3
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Transform ---
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

# --- Build Vocabulary ---
with open("dataset/Train.json") as f:
    train_data = json.load(f)["annotations"]
captions = [
    " ".join(item[0]["storytext"] for item in group)
    for group in json.load(open("dataset/Train.json"))["annotations"]
    if len(group) == 5
]
vocab = Vocabulary(FREQ_THRESHOLD)
vocab.build_vocab(captions)

# --- Datasets ---
train_ds = StoryDataset("dataset/Train.json", "dataset/Images", vocab, transform)
val_ds = StoryDataset("dataset/Validation.json", "dataset/Images", vocab, transform)

train_dl = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: x
)
val_dl = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: x
)

# --- Model ---
encoder = CNNEncoder(EMBED_SIZE).to(DEVICE)
decoder = LSTMDecoder(EMBED_SIZE, HIDDEN_SIZE, len(vocab)).to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx["<PAD>"])
optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()), lr=LR
)

checkpoint_path = "checkpoints/story_model.pth"
if os.path.exists(checkpoint_path):
    last_epoch, last_train_loss, last_val_loss = load_checkpoint(
        checkpoint_path, encoder, decoder, optimizer
    )
    print(
        f"Resumed from Epoch {last_epoch} | Train Loss: {last_train_loss:.4f}, Val Loss: {last_val_loss:.4f}"
    )
else:
    print("No checkpoint found. Starting from scratch.")

# --- Training ---
train_losses, val_losses = [], []
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
    encoder.train()
    decoder.train()
    total_loss = 0

    for batch in tqdm(train_dl, desc=f"Train Epoch {epoch+1}"):
        images, captions = zip(*batch)
        images = torch.stack(images).to(DEVICE)  # (B, 5, 3, H, W)
        captions = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True).to(
            DEVICE
        )  # (B, seq_len)

        inputs = captions[:, :-1]
        targets = captions[:, 1:]

        optimizer.zero_grad()
        features = encoder(images)  # (B, 5, embed)
        outputs = decoder(features, inputs)  # (B, seq_len-1, vocab_size)

        loss = criterion(outputs.view(-1, outputs.size(2)), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_losses.append(total_loss / len(train_dl))

    # --- Validation ---
    encoder.eval()
    decoder.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_dl, desc=f"Val Epoch {epoch+1}"):
            images, captions = zip(*batch)
            images = torch.stack(images).to(DEVICE)
            captions = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True).to(
                DEVICE
            )

            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            features = encoder(images)
            outputs = decoder(features, inputs)

            loss = criterion(outputs.view(-1, outputs.size(2)), targets.reshape(-1))
            val_loss += loss.item()

    val_losses.append(val_loss / len(val_dl))
    print(f"Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")

    # Save best model
    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        save_checkpoint(
            encoder,
            decoder,
            optimizer,
            epoch + 1,
            train_losses[-1],
            val_losses[-1],
            path=checkpoint_path,
        )

# --- Plot ---
plot_losses(train_losses, val_losses)
