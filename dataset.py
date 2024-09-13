# from torch.utils.data import DataLoader
# from datasets import load_dataset
# from torchvision import transforms
# from PIL import Image
# import torch
# from torchvision import transforms
# from PIL import Image
# # Define a custom dataset class for Flickr30k
# class Flickr30kDataset (torch.utils.data.Dataset):
#     def __init__(self):
#         self.dataset = load_dataset("nlphuji/flickr30k", cache_dir="./huggingface_data")
#         self.transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#         ])
#         self.cap_per_image = 2
#
#     def __len__(self):
#         return self.dataset.num_rows["test"] * self.cap_per_image
#
#     def __getitem__(self, idx):
#         original_idx = idx // self.cap_per_image
#         image = self.dataset["test"] [original_idx]["image"].convert("RGB")
#         image = self.transform(image)
#         # labels
#         caption = self.dataset["test"] [original_idx]["caption"] [idx % self.cap_per_image]
#         return {"image": image, "caption": caption}
#     # Create an instance of the custom dataset
#
# flickr30k_custom_dataset = Flickr30kDataset()

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch
import os
import csv


# Define a custom dataset class for Flickr30k with flickr30k-cna data
class Flickr30kCNADataset(Dataset):
    def __init__(self, image_dir, caption_file, transform=None):
        self.image_dir = image_dir
        self.caption_file = caption_file
        self.transform = transform

        # Load the captions
        self.captions = {}
        with open(caption_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                image_id = row[0]
                caption = row[1]
                if image_id not in self.captions:
                    self.captions[image_id] = []
                self.captions[image_id].append(caption)

        # Get the list of image IDs
        self.image_ids = list(self.captions.keys())

    def __len__(self):
        # Calculate the total number of image-caption pairs
        return sum(len(captions) for captions in self.captions.values())

    def __getitem__(self, idx):
        # Find the image ID and caption index for the given global index
        for image_id, captions in self.captions.items():
            if idx < len(captions):
                break
            idx -= len(captions)

        # Load the image
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Get the corresponding caption
        caption = captions[idx]
        return {"image": image, "caption": caption}

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# Create an instance of the custom dataset
image_dir = "flickr30k-images"  # Replace with the path to your flickr30k-image directory
caption_file = "flickr30k/flickr30k_cna_train.token"  # Replace with the path to your results_20130124.token file
flickr30k_dataset = Flickr30kCNADataset(image_dir=image_dir, caption_file=caption_file, transform=transform)

bili=0.6
# Create an instance of the custom dataset
image_dir = "flickr30k-images"  # Replace with the path to your flickr30k-image directory
caption_file1 = "flickr30k/flickr30k_cna_val.token"  # Replace with the path to your results_20130124.token file
flickr30k_dataset1 = Flickr30kCNADataset(image_dir=image_dir, caption_file=caption_file1, transform=transform)