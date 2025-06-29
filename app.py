# app.py

import torch
import torchvision.transforms as transforms
from PIL import Image
from models.encoder import CNNEncoder
from models.decoder import LSTMDecoder
from utils.vocab import Vocabulary
from utils.checkpoint import load_checkpoint
import json
import matplotlib.pyplot as plt
import textwrap

# -------- Config --------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_SIZE = 256
HIDDEN_SIZE = 512
FREQ_THRESHOLD = 3
CHECKPOINT_PATH = "checkpoints/story_model.pth"

# -------- Hardcoded Image Paths (change these) --------
image_paths = [
    "dataset/Images/2036.jpg",
    "dataset/Images/2037.jpg",
    "dataset/Images/2038.jpg",
    "dataset/Images/2039.jpg",
    "dataset/Images/2040.jpg",
]

# -------- Preprocessing --------
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

# -------- Load Vocabulary --------
with open("dataset/Train.json", "r") as f:
    annotations = json.load(f)["annotations"]
captions = [a[0]["storytext"] for a in annotations]
vocab = Vocabulary(freq_threshold=FREQ_THRESHOLD)
vocab.build_vocab(captions)

# -------- Load Model --------
encoder = CNNEncoder(EMBED_SIZE).to(DEVICE)
decoder = LSTMDecoder(EMBED_SIZE, HIDDEN_SIZE, len(vocab)).to(DEVICE)
load_checkpoint(CHECKPOINT_PATH, encoder, decoder)

encoder.eval()
decoder.eval()


# -------- Greedy Caption Generator --------
def generate_caption(encoder, decoder, images, vocab, max_len=100):
    with torch.no_grad():
        images = torch.stack(
            [transform(Image.open(img).convert("RGB")) for img in images]
        )
        images = images.unsqueeze(0).to(DEVICE)  # (1, 5, 3, H, W)

        features = encoder(images)  # (1, 5, embed)
        combined = features.view(1, -1)
        feat_embed = decoder.feat_proj(combined).unsqueeze(1)  # (1, 1, embed)

        inputs = [vocab.word2idx["<SOS>"]]
        story_caption = []

        for _ in range(max_len):
            input_tensor = torch.tensor(inputs).unsqueeze(0).to(DEVICE)
            embedded = decoder.embedding(input_tensor)
            repeated_feat = feat_embed.expand(-1, embedded.size(1), -1)
            lstm_input = torch.cat([embedded, repeated_feat], dim=2)

            out, _ = decoder.lstm(lstm_input)
            out = decoder.linear(out[:, -1, :])
            predicted = out.argmax(1).item()

            if predicted == vocab.word2idx["<EOS>"]:
                break

            story_caption.append(vocab.idx2word[predicted])
            inputs.append(predicted)

        return " ".join(story_caption)


# -------- Run Inference --------
fig, axes = plt.subplots(1, 5, figsize=(20, 5))

story = generate_caption(encoder, decoder, image_paths, vocab)
print("Generated story:", story)

# Show each image
for i, img_path in enumerate(image_paths):
    image = Image.open(img_path).convert("RGB")
    axes[i].imshow(image)
    axes[i].axis("off")

# Show story above the whole row
wrapped_story = "\n".join(textwrap.wrap(story, width=120))
fig.suptitle(wrapped_story, fontsize=12, y=1.05)

plt.tight_layout()
plt.show()
