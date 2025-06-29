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

        # Group by story_id
        story_groups = {}
        for item in data:
            item = item[0]
            sid = item["story_id"]
            story_groups.setdefault(sid, []).append(item)

        # Filter and sort stories
        for group in story_groups.values():
            if len(group) != 5:
                continue
            group = sorted(group, key=lambda x: x["image_order"])
            image_ids = [int(x["youtube_image_id"]) for x in group]
            full_caption = " ".join([x["storytext"] for x in group])
            self.sequences.append((image_ids, full_caption))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        image_ids, full_caption = self.sequences[idx]
        images = []

        for img_id in image_ids:
            img_path = os.path.join(self.image_dir, f"{img_id}.jpg")
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            images.append(image)

        # Tokenize full story caption
        tokens = (
            [self.vocab.word2idx["<SOS>"]]
            + self.vocab.numericalize(full_caption)
            + [self.vocab.word2idx["<EOS>"]]
        )
        caption_tensor = torch.tensor(tokens, dtype=torch.long)

        return torch.stack(images), caption_tensor  # (5, 3, H, W), (seq_len,)
