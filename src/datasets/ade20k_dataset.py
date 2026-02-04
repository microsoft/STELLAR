import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import kornia.augmentation as K
import numpy as np

# --- Define data augmentations ---
random_color_blur = torch.nn.Sequential(
    K.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.2),
    K.RandomGrayscale(p=0.2),
    K.RandomGaussianBlur((5, 5), (0.1, 1.0), p=0.2),
)

train_transform = K.AugmentationSequential(
    K.RandomResizedCrop(size=(224, 224), scale=(0.25, 1.0), resample='bicubic'),
    K.RandomHorizontalFlip(p=0.5),
    data_keys=["input", "mask"],
)

val_transform = K.AugmentationSequential(
    K.Resize(size=256, resample='bicubic'),
    K.CenterCrop(size=224),
    data_keys=["input", "mask"],
)

local_transform = torch.nn.Sequential(
    K.RandomResizedCrop(size=(96, 96), scale=(0.2, 0.6), resample='bicubic'),
    K.RandomHorizontalFlip(p=0.5),
    random_color_blur,
)


# --- ADE20K Dataset Class ---
class ADE20KSegmentationDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="training",  # 'training' or 'validation'
        pretraining=False,
        aug_input=False,
        crop_size=(224, 224),
    ):
        assert split in [
            "training",
            "validation",
        ], "split must be 'training' or 'validation'"
        self.pretraining = pretraining
        self.aug_input = aug_input
        self.crop_size = crop_size
        self.split = split

        self.image_dir = os.path.join(root_dir, "images", split)
        self.mask_dir = os.path.join(root_dir, "annotations", split)

        self.image_files = sorted(
            [f for f in os.listdir(self.image_dir) if f.endswith(".jpg")]
        )
        self.mask_files = sorted(
            [f for f in os.listdir(self.mask_dir) if f.endswith(".png")]
        )

        assert len(self.image_files) == len(
            self.mask_files
        ), "Number of images and masks do not match"

        self.transform = train_transform if split == "training" else val_transform

    def __len__(self):
        # return 100  # For testing purposes, return a fixed length
        # In practice, this should return the actual number of samples
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        image = TF.to_tensor(image)
        mask = torch.from_numpy(np.array(mask)).long()

        # Adjust label IDs: ADE20K uses 1–150 → shift to 0–149, and set 0 → 255
        mask = mask - 1
        mask[mask == -1] = 255  # originally 0, now marked as ignore
        mask[mask >= 150] = 255  # just in case, out-of-bound check

        if self.pretraining:
            image_views = image.unsqueeze(0).expand(4, -1, -1, -1)
            x0 = train_transform(image_views)
            x_global = random_color_blur(x0)
            x_local = local_transform(image_views)
            x_in = x_global[0] if self.aug_input else x0[0]
            return {
                "image": x_in,
                "global_view": x_global,
                "local_view": x_local,
                "labels": mask,
                "filename": img_path,
            }
        else:
            # Add batch dimension for Kornia
            image = image.unsqueeze(0)  # [1, C, H, W]
            mask = mask.unsqueeze(0)  # [1, H, W]

            image, mask = self.transform(image, mask)

            image = image.squeeze(0)  # [C, H, W]
            mask = mask.squeeze(0).squeeze(0).long()  # [H, W]

            return {
                "image": image.clamp(0, 1),
                "labels": mask,
                "filename": img_path,
            }
