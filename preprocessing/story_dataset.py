# preprocessing/story_dataset.py
import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset


class StoryDataset(Dataset):
    def __init__(self, json_path, image_dir, vocab, transform=None):
        self.image_dir = image_dir
        self.vocab = vocab
        self.transform = transform
        self.sequences = []

        with open(json_path, "r") as f:
            data = json.load(f)["annotations"]

        story_groups = {}
        for item in data:
            item = item[0]
            sid = item["story_id"]
            story_groups.setdefault(sid, []).append(item)

        for group in story_groups.values():
            if len(group) != 5:
                continue
            group = sorted(group, key=lambda x: x["image_order"])
            image_ids = [int(x["youtube_image_id"]) for x in group]
            captions = [x["storytext"] for x in group]
            self.sequences.append((image_ids, captions))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        image_ids, captions = self.sequences[idx]
        images, tokenized = [], []
        for i, img_id in enumerate(image_ids):
            img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            images.append(image)

            tokens = (
                [self.vocab.word2idx["<SOS>"]]
                + self.vocab.numericalize(captions[i])
                + [self.vocab.word2idx["<EOS>"]]
            )
            tokenized.append(torch.tensor(tokens, dtype=torch.long))

        return torch.stack(images), tokenized  # (5, 3, H, W), list of token tensors
