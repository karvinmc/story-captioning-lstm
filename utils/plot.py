# utils/plot.py
import matplotlib.pyplot as plt


def plot_losses(train, val):
    plt.plot(train, label="Train Loss")
    plt.plot(val, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()
