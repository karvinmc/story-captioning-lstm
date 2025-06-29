import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from models.cnn import SimpleCNN
from models.lstm import CaptionLSTM
from models.story_encoder import StoryEncoder
from utils.vocab import Vocabulary
from config import *

# --- CONFIG ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_path = "checkpoints/cnn_model.pth"
lstm_path = "checkpoints/lstm_model.pth"
story_encoder_path = "checkpoints/story_encoder.pth"
image_paths = [
    "img/story1.jpg",
    "img/story2.jpg",
    "img/story3.jpg",
]  # ubah dengan gambar yang ingin diuji

# --- Load Vocabulary ---
torch.serialization.add_safe_globals([Vocabulary])
vocab = torch.load("checkpoints/vocab.pt")

# --- Load Models ---
cnn = SimpleCNN(output_dim=EMBEDDING_DIM).to(device)
story_encoder = StoryEncoder(feature_dim=EMBEDDING_DIM, hidden_dim=EMBEDDING_DIM).to(device)
lstm = CaptionLSTM(
    embed_size=EMBEDDING_DIM,
    hidden_size=HIDDEN_DIM,
    vocab_size=len(vocab.stoi),
    feature_size=EMBEDDING_DIM,
).to(device)

cnn.load_state_dict(torch.load(cnn_path, map_location=device))
lstm.load_state_dict(torch.load(lstm_path, map_location=device))
if torch.exists(story_encoder_path):
    story_encoder.load_state_dict(torch.load(story_encoder_path, map_location=device))
cnn.eval()
lstm.eval()
story_encoder.eval()

# --- Preprocess Image ---
transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])


def load_images(paths: list[str]):
    images = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        images.append(transform(image))
    return images


# --- Generate Caption Function ---
def generate_caption_from_sequence(image_tensors: list[torch.Tensor], max_len=30):
    with torch.no_grad():
        image_tensors = torch.stack(image_tensors).to(device)  # [N, 3, 128, 128]
        features = cnn(image_tensors)  # [N, feature_dim]
        features = features.unsqueeze(0)  # [1, N, feature_dim]
        story_feature = story_encoder(features)  # [1, feature_dim]

        caption = [vocab.stoi["<SOS>"]]
        for _ in range(max_len):
            input_seq = torch.tensor([caption], device=device)
            out = lstm(story_feature, input_seq)
            next_token = out[0, -1].argmax().item()
            caption.append(next_token)
            if next_token == vocab.stoi["<EOS>"]:
                break

        words = [vocab.itos[idx] for idx in caption[1:-1]]
        return " ".join(words)


def clean_caption(caption: str) -> str:
    # Split by '.' and take the first full sentence
    parts = caption.strip().split(".")
    if parts:
        first_sentence = parts[0].strip()
        if first_sentence:
            return first_sentence + "."  # add the period back
    return caption  # fallback if no period found


# --- Run Inference ---
if __name__ == "__main__":
    images = load_images(image_paths)
    caption = generate_caption_from_sequence(images)
    caption = clean_caption(
        caption
    )  # Clean caption to ensure it ends with a period

    print("Generated Story Caption:", caption)

    # Tampilkan semua gambar secara berurutan
    fig, axs = plt.subplots(1, len(images), figsize=(5 * len(images), 5))
    for i, (img, ax) in enumerate(zip(images, axs)):
        ax.imshow(img.permute(1, 2, 0))
        ax.set_title(f"Image {i+1}")
        ax.axis("off")
    plt.suptitle(caption)
    plt.show()
