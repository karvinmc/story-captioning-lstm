import torch
import torchvision.transforms as transforms
from PIL import Image
from models.encoder import CNNEncoder
from models.decoder import LSTMDecoder
from utils.vocab import Vocabulary
from utils.checkpoint import load_checkpoint
import matplotlib.pyplot as plt
import textwrap
import pickle
import os

# -------- Config --------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_SIZE = 256
HIDDEN_SIZE = 512
FREQ_THRESHOLD = 3
CHECKPOINT_PATH = "checkpoints/story_model.pth"
VOCAB_PATH = "checkpoints/vocab.pkl"

# -------- Hardcoded Image Paths --------
image_paths = [
    "dataset/Images/2036.jpg",
    "dataset/Images/2037.jpg",
    "dataset/Images/2038.jpg",
    "dataset/Images/2039.jpg",
    "dataset/Images/2040.jpg",
]

# -------- Safety Check --------
assert all(
    os.path.exists(path) for path in image_paths
), "One or more image paths do not exist!"
assert os.path.exists(CHECKPOINT_PATH), "Checkpoint file not found!"
assert os.path.exists(VOCAB_PATH), "Vocabulary file not found!"

# -------- Preprocessing --------
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

# -------- Load Vocabulary --------
with open(VOCAB_PATH, "rb") as f:
    vocab = pickle.load(f)

# -------- Load Model --------
encoder = CNNEncoder(EMBED_SIZE).to(DEVICE)
decoder = LSTMDecoder(EMBED_SIZE, HIDDEN_SIZE, len(vocab)).to(DEVICE)
load_checkpoint(CHECKPOINT_PATH, encoder, decoder)

encoder.eval()
decoder.eval()


# -------- Greedy Caption Generator --------
def generate_caption(encoder, decoder, images, vocab, max_len=100):
    with torch.no_grad():
        # Load and preprocess images
        image_tensors = [transform(Image.open(img).convert("RGB")) for img in images]
        images_batch = (
            torch.stack(image_tensors).unsqueeze(0).to(DEVICE)
        )  # (1, 5, 3, 224, 224)

        # Encode images
        features = encoder(images_batch)  # (1, 5, embed)
        combined = features.view(1, -1)  # (1, 5 * embed)
        feat_embed = decoder.feat_proj(combined).unsqueeze(1)  # (1, 1, embed)

        # Start decoding
        inputs = [vocab.word2idx["<SOS>"]]
        story_caption = []

        for _ in range(max_len):
            input_tensor = torch.tensor(inputs).unsqueeze(0).to(DEVICE)  # (1, seq_len)
            embedded = decoder.embedding(input_tensor)  # (1, seq_len, embed)
            repeated_feat = feat_embed.expand(
                -1, embedded.size(1), -1
            )  # (1, seq_len, embed)
            lstm_input = torch.cat(
                [embedded, repeated_feat], dim=2
            )  # (1, seq_len, embed*2)

            out, _ = decoder.lstm(lstm_input)
            out = decoder.linear(out[:, -1, :])  # (1, vocab_size)
            predicted = out.argmax(1).item()

            if predicted == vocab.word2idx["<EOS>"]:
                break

            story_caption.append(vocab.idx2word[predicted])
            inputs.append(predicted)

        return " ".join(story_caption)


# -------- Run Inference --------
story = generate_caption(encoder, decoder, image_paths, vocab)
print("Generated Story:\n", story)

# -------- Visualization --------
fig, axes = plt.subplots(1, 5, figsize=(20, 5))

for i, img_path in enumerate(image_paths):
    image = Image.open(img_path).convert("RGB")
    axes[i].imshow(image)
    axes[i].axis("off")

# Wrap long story text
wrapped_story = "\n".join(textwrap.wrap(story, width=120))
fig.suptitle(wrapped_story, fontsize=12, y=1.08)  # Adjust y if text overlaps

plt.tight_layout()
plt.show()
