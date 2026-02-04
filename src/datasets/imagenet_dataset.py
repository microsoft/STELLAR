import json
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import kornia.augmentation as K



class ImageNetKaggle(Dataset):
    def __init__(self, root, split="train", pretraining=True, resolution=224):
        self.pretraining = pretraining
        self.split = split
        self.samples = []
        self.targets = []
        # self.transform = train_transform if split == "train" else val_transform    # normal transform
        if split == "train":
            self.transform = torch.nn.Sequential(
                K.RandomResizedCrop(size=(resolution,resolution), scale=(0.20,1.0), resample='bicubic'),
                K.RandomHorizontalFlip(p=0.5),
            )
        else:
            crop_res = int(resolution * 256 / 224)
            self.transform = torch.nn.Sequential(
                K.Resize(size=crop_res, resample='bicubic'),
                K.CenterCrop(size=resolution),
            )
        self.global_transform = torch.nn.Sequential(
            K.RandomResizedCrop(size=(resolution,resolution), scale=(0.36,1.0), resample='bicubic'),
            K.RandomHorizontalFlip(p=0.5),
        )
        self.local_transform = torch.nn.Sequential(
            K.RandomResizedCrop(size=(96,96), scale=(0.06, 0.36), resample='bicubic'),
            K.RandomHorizontalFlip(p=0.5),
        )
        self.syn_to_class = {}
        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
                    json_file = json.load(f)
                    for class_id, v in json_file.items():
                        self.syn_to_class[v[0]] = int(class_id)
        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
                    self.val_to_syn = json.load(f)
        samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)
        for entry in os.listdir(samples_dir):
            if split == "train":
                syn_id = entry
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)
            elif split == "val":
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)
    def __len__(self):
            return len(self.samples)
    def __getitem__(self, idx):
            img = Image.open(self.samples[idx]).convert("RGB")
            img = transforms.ToTensor()(img)  # Convert PIL image to tensor
            if self.pretraining:
                x = self.transform(img).clamp(0.0, 1.0)  # Apply the normal transform
                img_views = img.unsqueeze(0).expand(8, -1, -1, -1)
                global_views = self.global_transform(img_views).clamp(0.0, 1.0)  # Apply the normal transform to all views
                local_views = self.local_transform(img_views).clamp(0.0, 1.0)

                return {
                    "image": global_views[0],
                    "global_views": global_views,
                    "local_views": local_views,
                    "labels": global_views[0],
                    "filename": self.samples[idx],
                }
            else:
                x = self.transform(img).clamp(0.0, 1.0)

                return {
                    "image": x[0],
                    "labels": self.targets[idx],
                    "filename": self.samples[idx],
                }
