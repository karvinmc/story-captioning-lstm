import torch
import torchvision.transforms as transforms
from PIL import Image
from models.clip_transformer import CLIPTransformerModel
from utils.vocab import Vocabulary
from utils.checkpoint import load_checkpoint
import matplotlib.pyplot as plt
import textwrap
import os
import pickle

# -------- Config --------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_SIZE = 768  # Match CLIP's feature dimension
FREQ_THRESHOLD = 1
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
assert all(os.path.exists(path) for path in image_paths), "One or more image paths do not exist!"
assert os.path.exists(VOCAB_PATH), "Vocabulary file not found!"

# -------- Preprocessing --------
transform = transforms.Compose([
    transforms.Resize((336, 336)),  # CLIP requires 336x336
    transforms.ToTensor()
])

# -------- Load Vocabulary --------
with open(VOCAB_PATH, "rb") as f:
    vocab = pickle.load(f)

# -------- Load Model --------
model = CLIPTransformerModel(len(vocab), embed_size=EMBED_SIZE).to(DEVICE)
if os.path.exists(CHECKPOINT_PATH):
    load_checkpoint(CHECKPOINT_PATH, model)
else:
    print("Checkpoint not found. Using untrained model (results may be poor).")

model.eval()

# -------- Generate Caption --------
with torch.no_grad():
    image_tensors = [transform(Image.open(img).convert("RGB")) for img in image_paths]
    images_batch = torch.stack(image_tensors).unsqueeze(0).to(DEVICE)  # (1, 5, 3, 336, 336)
    generated = model.generate(
        images_batch,
        max_len=100,
        start_token_idx=vocab.word2idx["<SOS>"],
        end_token_idx=vocab.word2idx["<EOS>"],
        beam_width=3
    )
    story = " ".join([vocab.idx2word[idx.item()] for idx in generated[0] if idx.item() not in [vocab.word2idx["<SOS>"], vocab.word2idx["<EOS>"]]])

print("Generated Story:\n", story)

# -------- Visualization --------
num_images = len(image_paths)
fig, axes = plt.subplots(1, num_images, figsize=(4 * num_images, 5))
if num_images == 1:
    axes = [axes]
for i, img_path in enumerate(image_paths):
    image = Image.open(img_path).convert("RGB")
    axes[i].imshow(image)
    axes[i].axis("off")

wrapped_story = "\n".join(textwrap.wrap(story, width=120))
fig.suptitle(wrapped_story, fontsize=12, y=1.08)

plt.tight_layout()
plt.show()