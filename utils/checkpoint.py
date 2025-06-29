# utils/checkpoint.py

import torch


def save_checkpoint(
    encoder,
    decoder,
    optimizer,
    epoch,
    train_loss,
    val_loss,
    path="checkpoints/story_model.pth",
):
    checkpoint = {
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, encoder, decoder, optimizer=None):
    checkpoint = torch.load(path, map_location="cpu")
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"[✓] Loaded checkpoint from {path}")
    return checkpoint["epoch"], checkpoint["train_loss"], checkpoint["val_loss"]
