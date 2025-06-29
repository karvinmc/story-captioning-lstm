import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
import os
import json

from models.cnn import SimpleCNN
from models.lstm import CaptionLSTM
from models.story_encoder import StoryEncoder
from utils.vocab import Vocabulary
from utils.preprocessing import StoryDataset
from config import *

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    train_json = "dataset/Train.json"
    with open(train_json, "r") as f:
        data = json.load(f)
        all_captions = []
        for storylet in data["annotations"]:
            for item in storylet:
                all_captions.append(item["storytext"])

    vocab = Vocabulary(freq_threshold=VOCAB_THRESHOLD)
    vocab.build_vocab(all_captions)

    torch.save(vocab, "checkpoints/vocab.pt")

    # Dataset & DataLoader
    dataset = StoryDataset("dataset/Images", train_json, vocab)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )

    # Validation Dataset & DataLoader
    val_json = "dataset/Validation.json"
    val_dataset = StoryDataset("dataset/Images", val_json, vocab)
    val_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # Model
    cnn = SimpleCNN(output_dim=EMBEDDING_DIM).to(device)
    story_encoder = StoryEncoder(
        feature_dim=EMBEDDING_DIM, hidden_dim=EMBEDDING_DIM, dropout=0.3
    ).to(device)
    lstm = CaptionLSTM(
        embed_size=EMBEDDING_DIM,
        hidden_size=HIDDEN_DIM,
        vocab_size=len(vocab.stoi),
        feature_size=EMBEDDING_DIM,  # Ensure this matches CNN output
        dropout=0.3
    ).to(device)

    # Load weights if available
    if os.path.exists("checkpoints/cnn_model.pth"):
        cnn.load_state_dict(torch.load("cnn_model.pth", map_location=device))
        print("Loaded CNN weights from cnn_model.pth")
    if os.path.exists("checkpoints/story_encoder.pth"):
        story_encoder.load_state_dict(
            torch.load("checkpoints/story_encoder.pth", map_location=device)
        )
        print("Loaded StoryEncoder weights from story_encoder.pth")
    if os.path.exists("checkpoints/lstm_model.pth"):
        lstm.load_state_dict(torch.load("lstm_model.pth", map_location=device))
        print("Loaded LSTM weights from lstm_model.pth")

    # Training Components
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
    params = (
        list(cnn.parameters())
        + list(story_encoder.parameters())
        + list(lstm.parameters())
    )
    optimizer = optim.Adam(params, lr=3e-4, weight_decay=1e-5)

    def evaluate(model_cnn, model_story_encoder, model_lstm, dataloader, criterion):
        model_cnn.eval()
        model_story_encoder.eval()
        model_lstm.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, caps in dataloader:
                B, N, C, H, W = imgs.shape
                imgs = imgs.view(B * N, C, H, W).to(device)
                caps = caps.to(device)
                features = model_cnn(imgs)
                features = features.view(B, N, -1)
                story_feature = model_story_encoder(features)
                outputs = model_lstm(story_feature, caps[:, :-1])
                loss = criterion(
                    outputs.reshape(-1, outputs.shape[2]), caps[:, 1:].reshape(-1)
                )
                val_loss += loss.item()
        return val_loss / len(dataloader)

    best_val_loss = float("inf")
    patience = 3
    counter = 0

    train_losses = []
    val_losses = []

    # Training Loop
    EPOCHS = 50
    for epoch in range(EPOCHS):
        cnn.train()
        story_encoder.train()
        lstm.train()
        epoch_loss = 0

        for imgs, caps in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            # imgs: [B, N, 3, H, W] (N = number of images per story)
            B, N, C, H, W = imgs.shape
            imgs = imgs.view(B * N, C, H, W).to(device)
            caps = caps.to(device)
            optimizer.zero_grad()

            features = cnn(imgs)  # [B*N, feature_dim]
            features = features.view(B, N, -1)  # [B, N, feature_dim]
            story_feature = story_encoder(features)  # [B, feature_dim]

            outputs = lstm(story_feature, caps[:, :-1])  # input
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), caps[:, 1:].reshape(-1)
            )  # target
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(dataloader)
        val_loss = evaluate(cnn, story_encoder, lstm, val_dataloader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(
            f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        # Early stopping (sementara tak disable)
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     counter = 0
        #     # Save best model
        #     torch.save(cnn.state_dict(), "checkpoints/cnn_model.pth")
        #     torch.save(story_encoder.state_dict(), "checkpoints/story_encoder.pth")
        #     torch.save(lstm.state_dict(), "checkpoints/lstm_model.pth")
        # else:
        #     counter += 1
        #     if counter >= patience:
        #         print("Early stopping triggered.")
        #         break

    # Save models after training
    torch.save(cnn.state_dict(), "checkpoints/cnn_model.pth")
    torch.save(story_encoder.state_dict(), "checkpoints/story_encoder.pth")
    torch.save(lstm.state_dict(), "checkpoints/lstm_model.pth")

    # Plot training and validation loss curves
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.show()
