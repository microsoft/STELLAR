import torch
import torch.nn as nn
import torch.nn.functional as F


def get_norm(norm_type, num_channels):
    if norm_type == "group":
        return nn.GroupNorm(1, num_channels)
    elif norm_type == "batch":
        return nn.BatchNorm2d(num_channels)
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")


def reshape_feature_map_for_spatial_tasks(features, key=None):
    if features.ndim == 4:
        return features
    if features.ndim == 3:
        B, N, C = features.shape
        H = W = int(N**0.5)
        if H * W != N:
            raise ValueError(
                f"Cannot reshape sequence of length {N} to square grid for key '{key}'"
            )
        return features.permute(0, 2, 1).reshape(B, C, H, W)
    raise ValueError(f"Unsupported feature shape {features.shape} for key '{key}'")


class SegmentationProbing(nn.Module):
    def __init__(
        self,
        model_backbone,
        is_baseline=False,
        feature_key="dense",
        feature_dim=768,
        num_classes=150,
        freeze_backbone=True,
        freeze_model=False,
        model_checkpoint=None,
        resize_output: tuple = None,
    ):
        super().__init__()
        self.model_backbone = model_backbone
        self.is_baseline = is_baseline
        self.feature_key = feature_key
        self.freeze_backbone = freeze_backbone
        self.freeze_model = freeze_model
        self.resize_output = resize_output
        
        if self.is_baseline:
            feature_dim = self.model_backbone.dim

        self.decoder_head = nn.Sequential(
            get_norm("batch", feature_dim),
            nn.Conv2d(feature_dim, num_classes, kernel_size=1),
        )

        if model_checkpoint:
            checkpoint = torch.load(model_checkpoint, map_location="cpu")
            state_dict = {
                k[6:] if k.startswith("model.") else k: v
                for k, v in checkpoint["state_dict"].items()
            }
            self.model_backbone.load_state_dict(state_dict, strict=True)
            print("Backbone checkpoint loaded successfully!")

        if self.freeze_backbone:
            self.model_backbone.eval()
            for param in self.model_backbone.parameters():
                param.requires_grad = False

        if self.freeze_model:
            self.eval()
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, inputs: dict):

        if self.freeze_backbone:
            with torch.no_grad():
                if self.is_baseline:
                    predictions = self.model_backbone(inputs['image'])
                else:
                    predictions = self.model_backbone(inputs, mode="eval")["predictions"]
        else:
            if self.is_baseline:
                predictions = self.model_backbone(inputs['image'])
            else:
                predictions = self.model_backbone(inputs, mode="eval")["predictions"]

        features = predictions[self.feature_key]
        features_2d = reshape_feature_map_for_spatial_tasks(
            features, key=self.feature_key
        )
        logits = self.decoder_head(features_2d)

        if self.resize_output is not None:
            logits = F.interpolate(
                logits, size=self.resize_output, mode="bilinear", align_corners=False
            )

        return {
            "predictions": logits,
            self.feature_key: features_2d,
        }
        
        
class MultiLayerSegmentationProbing(nn.Module):
    def __init__(
        self,
        model_backbone,
        feature_keys=("dense", "feature", "map"),
        feature_dims={"dense": 768, "feature": 768},
        num_classes=150,
        inter_dim=256,
        freeze_backbone=True,
        freeze_model=False,
        model_checkpoint=None,
        resize_output: tuple = None,
    ):
        super().__init__()
        self.model_backbone = model_backbone
        self.feature_keys = feature_keys
        self.freeze_backbone = freeze_backbone
        self.freeze_model = freeze_model
        self.resize_output = resize_output

        self.feature_proj = nn.ModuleList()
        for key in self.feature_keys:
            in_dim = feature_dims[key]
            self.feature_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_dim, inter_dim, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(inter_dim),
                )
            )

        self.fusion = nn.Sequential(
            nn.Conv2d(inter_dim * len(self.feature_keys), inter_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_dim, num_classes, 1),
        )

        if model_checkpoint:
            checkpoint = torch.load(model_checkpoint, map_location="cpu")
            state_dict = {
                k[6:] if k.startswith("model.") else k: v
                for k, v in checkpoint["state_dict"].items()
            }
            self.model_backbone.load_state_dict(state_dict, strict=True)
            print("Backbone checkpoint loaded successfully!")

        if self.freeze_backbone:
            self.model_backbone.eval()
            for param in self.model_backbone.parameters():
                param.requires_grad = False

        if self.freeze_model:
            self.eval()
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, inputs: dict):
        image = inputs["image"]
        orig_size = image.shape[-2:]

        if self.freeze_backbone:
            with torch.no_grad():
                predictions = self.model_backbone(inputs, mode="eval")["predictions"]
        else:
            predictions = self.model_backbone(inputs, mode="train")["predictions"]

        features = []
        for key, proj in zip(self.feature_keys, self.feature_proj):
            feat = predictions[key]
            feat = reshape_feature_map_for_spatial_tasks(feat, key)
            feat = proj(feat)
            features.append(feat)

        fused = torch.cat(features, dim=1)
        logits = self.fusion(fused)

        if self.resize_output is not None:
            logits = F.interpolate(
                logits, size=self.resize_output, mode="bilinear", align_corners=False
            )

        logits = F.interpolate(
            logits, size=orig_size, mode="bilinear", align_corners=False
        )

        return {
            "predictions": logits,
            "features": fused,
        }
