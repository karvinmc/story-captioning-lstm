import os
import torch
import json
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from collections import defaultdict
from config import MAX_CAPTION_LEN


class StoryDataset(Dataset):
    def __init__(
        self, image_folder, json_file, vocab, transform=None, max_len=MAX_CAPTION_LEN
    ):
        self.image_folder = image_folder
        self.vocab = vocab
        self.max_len = max_len
        self.transform = transform or transforms.Compose(
            [transforms.Resize((128, 128)), transforms.ToTensor()]
        )

        # Load JSON file
        with open(json_file, "r") as f:
            data = json.load(f)

        # Group data by story_id
        self.stories = defaultdict(list)
        for item_list in data["annotations"]:
            for item in item_list:
                self.stories[item["story_id"]].append(item)

        # Build entries: list of (image_paths, full_story_text)
        self.entries = []
        for story_id, story_items in self.stories.items():
            sorted_items = sorted(story_items, key=lambda x: x["image_order"])
            image_paths = [
                os.path.join(self.image_folder, f'{int(i["youtube_image_id"])}.jpg')
                for i in sorted_items
            ]
            story_text = " ".join([i["storytext"] for i in sorted_items])
            self.entries.append((image_paths, story_text))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        image_paths, caption = self.entries[idx]
        images = []

        for path in image_paths:
            image = Image.open(path).convert("RGB")
            image = self.transform(image)
            images.append(image)

        # Convert caption to tokens
        tokens = [self.vocab.stoi["<SOS>"]]
        tokens += self.vocab.numericalize(caption)[: self.max_len - 2]
        tokens += [self.vocab.stoi["<EOS>"]]
        tokens += [self.vocab.stoi["<PAD>"]] * (self.max_len - len(tokens))

        images_tensor = torch.stack(images)  # [N, 3, H, W]
        return images_tensor, torch.tensor(tokens)
