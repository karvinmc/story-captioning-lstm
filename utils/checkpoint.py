# utils/checkpoint.py

import torch


def save_checkpoint(
    encoder,
    decoder,
    optimizer,
    epoch,
    train_loss,
    val_loss,
    path,
    train_losses=None,
    val_losses=None,
):
    torch.save(
        {
            "epoch": epoch,
            "encoder_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
        },
        path,
    )


def load_checkpoint(path, encoder, decoder, optimizer=None):
    checkpoint = torch.load(path)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    print(f"[✓] Loaded checkpoint from {path}")
    return epoch, train_loss, val_loss, train_losses, val_losses
