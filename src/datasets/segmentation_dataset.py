import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import kornia.augmentation as K
from torchvision import datasets as tvd


data_splits = {
    "VOCSegmentation": {'train': 'train', 'test': 'val'},
    "Cityscapes": {'train': 'train', 'test': 'val'},
}


class SegmentationDataset(Dataset):
    def __init__(self, root, name, split="train", pretraining=False, resolution=224, crop_scale=0.5):
        self.pretraining = pretraining
        self.split = split
        self.name = name
        # self.transform = train_transform if split == "train" else val_transform    # normal transform
        if split == "train":
            self.transform = K.AugmentationSequential(
                K.RandomResizedCrop(size=(resolution,resolution), scale=(crop_scale,1.0), resample='bicubic'),
                K.RandomHorizontalFlip(p=0.5),
                data_keys=["input", "mask"],
            )
        else:
            crop_res = int(resolution * 256 / 224)
            self.transform = K.AugmentationSequential(
                K.Resize(size=crop_res, resample='bicubic'),
                K.CenterCrop(size=resolution),
                data_keys=["input", "mask"],
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
        if name == "VOCSegmentation":
            self.data = tvd.VOCSegmentation(os.path.join(root, name), image_set=split_name)
        elif name == "Cityscapes":
            self.data = tvd.Cityscapes(os.path.join(root, name), split=split_name, mode='fine', target_type='semantic')
        else:
            raise NotImplementedError(f"Dataset {name} not implemented.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, mask = self.data[idx]
        img = img.convert("RGB")
        img = transforms.ToTensor()(img)  # Convert PIL image to tensor
        mask = torch.from_numpy(np.array(mask)).long()
        # Adjust label IDs to start from 0
        mask = mask - 1
        mask[mask == -1] = 255  # originally 0, now marked as ignore
        mask[mask >= 150] = 255  # just in case, out-of-bound check
        if self.pretraining:
            x, label = self.transform(img, mask)  # Apply the normal transform
            img_views = img.unsqueeze(0).expand(8, -1, -1, -1)
            global_views = self.global_transform(img_views).clamp(0.0, 1.0)  # Apply the normal transform to all views
            local_views = self.local_transform(img_views).clamp(0.0, 1.0)

            return {
                "image": global_views[0],
                "image_probing": x[0].clamp(0.0, 1.0),
                "global_views": global_views,
                "local_views": local_views,
                "labels": label[0].clamp(0.0, 1.0),
                "filename": idx,
            }
        else:
            image, mask = self.transform(img, mask)

            image = image.squeeze(0)  # [C, H, W]
            mask = mask.squeeze(0).squeeze(0).long()  # [H, W]

            return {
                "image": image.clamp(0, 1),
                "labels": mask,
                "filename": idx,
            }