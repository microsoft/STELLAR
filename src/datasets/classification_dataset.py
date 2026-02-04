import json
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import kornia.augmentation as K
from torchvision import datasets as tvd


data_splits = {
    "Places365": {'train': 'train-standard', 'test': 'val'},
    "Food101": {'train': 'train', 'test': 'test'},
    "OxfordIIITPet": {'train': 'trainval', 'test': 'test'},
}


class ClassificationDataset(Dataset):
    def __init__(self, root, name, split="train", pretraining=False, resolution=224, crop_scale=0.5):
        self.pretraining = pretraining
        self.split = split
        self.name = name
        # self.transform = train_transform if split == "train" else val_transform    # normal transform
        if split == "train":
            self.transform = torch.nn.Sequential(
                K.RandomResizedCrop(size=(resolution,resolution), scale=(crop_scale,1.0), resample='bicubic'),
                K.RandomHorizontalFlip(p=0.5),
            )
        else:
            crop_res = int(resolution * 256 / 224)
            self.transform = torch.nn.Sequential(
                K.Resize(size=crop_res, resample='bicubic'),
                K.CenterCrop(size=resolution),
            )
        if pretraining:
            self.global_transform = torch.nn.Sequential(
                K.RandomResizedCrop(size=(resolution,resolution), scale=(0.36,1.0), resample='bicubic'),
                K.RandomHorizontalFlip(p=0.5),
            )
            self.local_transform = torch.nn.Sequential(
                K.RandomResizedCrop(size=(96, 96), scale=(0.06, 0.36), resample='bicubic'),
                K.RandomHorizontalFlip(p=0.5),
            )
        if name in data_splits:
            split_name = data_splits[name][split]
        else:
            split_name = split
        if name == "Places365":
            self.data = tvd.Places365(os.path.join(root, name), split=split_name)
        elif name == "Food101":
            self.data = tvd.Food101(os.path.join(root, name), split=split_name)
        elif name == "OxfordIIITPet":
            self.data = tvd.OxfordIIITPet(os.path.join(root, name), split=split_name, target_types="category")
        else:
            raise NotImplementedError(f"Dataset {name} not implemented.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx]
        img = img.convert("RGB")
        img = transforms.ToTensor()(img)  # Convert PIL image to tensor
        if self.pretraining:
            x = self.transform(img).clamp(0.0, 1.0)  # Apply the normal transform
            img_views = img.unsqueeze(0).expand(8, -1, -1, -1)
            global_views = self.global_transform(img_views).clamp(0.0, 1.0)  # Apply the normal transform to all views
            local_views = self.local_transform(img_views).clamp(0.0, 1.0)

            return {
                "image": global_views[0],
                "image_probing": x[0],
                "global_views": global_views,
                "local_views": local_views,
                "labels": target,
                "filename": idx,
            }
        else:
            x = self.transform(img).clamp(0.0, 1.0)  # Apply the normal transform

            return {
                "image": x[0],
                "labels": target,
                "filename": idx,
            }