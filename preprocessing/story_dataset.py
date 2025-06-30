import json
from PIL import Image
import torch
from collections import defaultdict


class StoryDataset:
    def __init__(self, json_path, img_dir, vocab, transform):
        self.img_dir = img_dir
        self.vocab = vocab
        self.transform = transform
        with open(json_path) as f:
            self.data = json.load(f)["annotations"]

        # Group annotations by story_id
        self.story_groups = defaultdict(list)
        for group in self.data:
            for item in group:
                self.story_groups[item["story_id"]].append(item)

        # Create list of stories
        self.stories = []
        for story_id, items in self.story_groups.items():
            items = sorted(items, key=lambda x: x["image_order"])
            images = [
                f"{self.img_dir}/{item['youtube_image_id']}.jpg" for item in items
            ]
            caption = " ".join(
                item["storytext"] for item in items if "storytext" in item
            )
            self.stories.append((images, caption))

        # print(f"Dataset size: {len(self.stories)}")
        # print(f"Sample story: {self.stories[0]}")

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        images, caption = self.stories[idx]
        # Load and transform images
        image_tensors = [
            self.transform(Image.open(img).convert("RGB")) for img in images
        ]
        images = torch.stack(image_tensors)  # (num_images, 3, H, W)
        # Numericalize caption
        caption = (
            [self.vocab.word2idx["<SOS>"]]
            + self.vocab.numericalize(caption)
            + [self.vocab.word2idx["<EOS>"]]
        )
        caption = torch.tensor(caption)
        return images, caption
