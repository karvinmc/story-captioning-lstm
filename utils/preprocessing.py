import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils.vocab import Vocabulary


class Dataset(Dataset):
    def __init__(self, image_folder, caption_file, vocab, transform=None, max_len=20):
        self.image_folder = image_folder
        self.vocab = vocab
        self.max_len = max_len
        self.transform = transform or transforms.Compose(
            [transforms.Resize((128, 128)), transforms.ToTensor()]
        )
        # Load caption file
        with open(caption_file, "r") as f:
            lines = f.readlines()

        # Mapping: image_name → [caption1, caption2, ...]
        self.data = {}
        for line in lines:
            if "," in line:
                img, caption = line.strip().split(",", 1)
                img = img.split("#")[0]
                self.data.setdefault(img, []).append(caption)

        # Gabungkan 3-5 caption menjadi satu kalimat naratif (story-driven)
        self.entries = []
        for img, captions in self.data.items():
            if len(captions) >= 3:
                story = " ".join(captions[:3])  # bisa disesuaikan
                self.entries.append((img, story))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        img_name, caption = self.entries[idx]
        img_path = os.path.join(self.image_folder, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        tokens = [self.vocab.stoi["<SOS>"]]
        tokens += self.vocab.numericalize(caption)[: self.max_len - 2]
        tokens += [self.vocab.stoi["<EOS>"]]
        tokens += [self.vocab.stoi["<PAD>"]] * (self.max_len - len(tokens))

        return image, torch.tensor(tokens)
