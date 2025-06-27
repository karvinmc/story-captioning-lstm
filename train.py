import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
import os

from models.cnn import SimpleCNN
from models.lstm import CaptionLSTM
from utils.vocab import Vocabulary
from utils.preprocessing import Dataset
from config import *

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    # Load caption & build vocab
    caption_file = "dataset/captions.txt"
    with open(caption_file, "r") as f:
        lines = f.readlines()[1:]  # skip header
        all_captions = [line.strip().split(",", 1)[1] for line in lines if "," in line]

    vocab = Vocabulary(freq_threshold=VOCAB_THRESHOLD)
    vocab.build_vocab(all_captions)

    # Dataset & DataLoader
    dataset = Dataset("dataset/Images", caption_file, vocab)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )

    # Model
    cnn = SimpleCNN(output_dim=EMBEDDING_DIM).to(device)
    lstm = CaptionLSTM(
        embed_size=EMBEDDING_DIM,
        hidden_size=HIDDEN_DIM,
        vocab_size=len(vocab.stoi),
        feature_size=EMBEDDING_DIM,  # Ensure this matches CNN output
    ).to(device)

    # Load weights if available
    if os.path.exists("cnn_model.pth"):
        cnn.load_state_dict(torch.load("cnn_model.pth", map_location=device))
        print("Loaded CNN weights from cnn_model.pth")
    if os.path.exists("lstm_model.pth"):
        lstm.load_state_dict(torch.load("lstm_model.pth", map_location=device))
        print("Loaded LSTM weights from lstm_model.pth")

    # Training Components
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
    params = list(cnn.parameters()) + list(lstm.parameters())
    optimizer = optim.Adam(params, lr=3e-4)

    # Training Loop
    EPOCHS = 10
    for epoch in range(EPOCHS):
        cnn.train()
        lstm.train()
        epoch_loss = 0

        for imgs, caps in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs, caps = imgs.to(device), caps.to(device)
            optimizer.zero_grad()

            features = cnn(imgs)
            outputs = lstm(features, caps[:, :-1])  # input
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), caps[:, 1:].reshape(-1)
            )  # target
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(dataloader):.4f}")

    # Save models after training
    torch.save(cnn.state_dict(), "cnn_model.pth")
    torch.save(lstm.state_dict(), "lstm_model.pth")