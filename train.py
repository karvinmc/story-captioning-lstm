import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from preprocessing.story_dataset import StoryDataset
from models.clip_transformer import CLIPTransformerModel
from utils.vocab import Vocabulary
from utils.plot import plot_losses
from utils.checkpoint import save_checkpoint, load_checkpoint
from collections import defaultdict
from tqdm import tqdm
import os
import json
import pickle

# --- Config ---
EPOCHS = 10
BATCH_SIZE = 4  # Reduced due to CLIP's memory requirements
EMBED_SIZE = 768  # Match CLIP-ViT-L-336px feature dimension
FREQ_THRESHOLD = 1
LR = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Transform ---
train_transform = transforms.Compose([
    transforms.Resize((336, 336)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor()
    # Jangan normalize! CLIPProcessor akan melakukan ini
])

val_transform = transforms.Compose([
    transforms.Resize((336, 336)),
    transforms.ToTensor()
    # Jangan normalize!
])

# --- Build Vocabulary ---
with open("dataset/Train.json") as f:
    train_data = json.load(f)
annotations = train_data["annotations"]

# Group annotations by story_id to form stories
story_groups = defaultdict(list)
for group in annotations:
    for item in group:
        story_id = item["story_id"]
        story_groups[story_id].append(item)

# Create captions by joining storytext for each story_id
captions = []
for story_id, items in story_groups.items():
    items = sorted(items, key=lambda x: x["image_order"])
    caption = " ".join(item["storytext"] for item in items if "storytext" in item)
    captions.append(caption)

vocab = Vocabulary(FREQ_THRESHOLD)
vocab.build_vocab(captions)

# Save vocab
os.makedirs("checkpoints", exist_ok=True)
with open("checkpoints/vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

print("Vocabulary saved to checkpoints/vocab.pkl")

# --- Datasets ---
train_ds = StoryDataset("dataset/Train.json", "dataset/Images", vocab, train_transform)
val_ds = StoryDataset("dataset/Validation.json", "dataset/Images", vocab, val_transform)

train_dl = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: x
)
val_dl = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: x
)

# --- Model ---
model = CLIPTransformerModel(len(vocab), embed_size=EMBED_SIZE).to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx["<PAD>"])
optimizer = torch.optim.Adam(
    list(model.transformer_decoder.parameters()) +
    list(model.embedding.parameters()) +
    list(model.linear.parameters()),
    lr=LR,
    weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

checkpoint_path = "checkpoints/story_model.pth"
start_epoch = 0
train_losses, val_losses = [], []
best_val_loss = float("inf")
patience = 5
counter = 0

if os.path.exists(checkpoint_path):
    last_epoch, last_train_loss, last_val_loss, train_losses, val_losses = load_checkpoint(
        checkpoint_path, model, optimizer
    )
    print(f"Resumed from Epoch {last_epoch} | Train Loss: {last_train_loss:.4f}, Val Loss: {last_val_loss:.4f}")
    start_epoch = last_epoch
else:
    print("No checkpoint found. Starting from scratch.")

# --- Training ---
for epoch in range(start_epoch, EPOCHS):
    print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
    model.train()
    total_loss = 0

    for batch in tqdm(train_dl, desc=f"Train Epoch {epoch+1}"):
        images, captions = zip(*batch)
        images = torch.stack(images).to(DEVICE)  # (B, 5, 3, 336, 336)
        captions = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True).to(DEVICE)  # (B, seq_len)

        inputs = captions[:, :-1]
        targets = captions[:, 1:]

        optimizer.zero_grad()
        # Generate causal mask for transformer
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(inputs.size(1)).to(DEVICE)
        outputs = model(images, inputs, tgt_mask=tgt_mask)  # (B, seq_len-1, vocab_size)

        loss = criterion(outputs.view(-1, outputs.size(2)), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_losses.append(total_loss / len(train_dl))

    # --- Validation ---
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_dl, desc=f"Val Epoch {epoch+1}"):
            images, captions = zip(*batch)
            images = torch.stack(images).to(DEVICE)
            captions = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True).to(DEVICE)

            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(inputs.size(1)).to(DEVICE)
            outputs = model(images, inputs, tgt_mask=tgt_mask)

            loss = criterion(outputs.view(-1, outputs.size(2)), targets.reshape(-1))
            val_loss += loss.item()

    val_losses.append(val_loss / len(val_dl))
    scheduler.step(val_losses[-1])
    print(f"Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")

    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        counter = 0
        save_checkpoint(
            model,
            optimizer,
            epoch + 1,
            train_losses[-1],
            val_losses[-1],
            path=checkpoint_path,
        )
    else:
        counter += 1
        print(f"Validation loss did not improve for {counter} epoch(s).")
        if counter >= patience:
            print("Early stopping triggered.")
            break

# --- Plot ---
plot_losses(train_losses, val_losses)