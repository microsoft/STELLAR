import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # logits: [B, C, H, W], targets: [B, H, W]
        valid_mask = targets != self.ignore_index
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, dtype=logits.dtype, device=logits.device)

        targets_clean = targets.clone()
        targets_clean[~valid_mask] = 0  # ensure valid one-hot

        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets_clean, num_classes=logits.shape[1])
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        valid_mask = valid_mask.unsqueeze(1)  # [B, 1, H, W]
        probs = probs * valid_mask
        targets_one_hot = targets_one_hot * valid_mask

        dims = (0, 2, 3)
        intersection = (probs * targets_one_hot).sum(dims)
        union = probs.sum(dims) + targets_one_hot.sum(dims)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class SegmentationHybridLoss(nn.Module):
    def __init__(
        self, ce_weight=1.0, dice_weight=1.0, edge_weight=0.0, ignore_index=255
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.edge_weight = edge_weight

        self.dice_loss = DiceLoss(ignore_index=ignore_index)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, predictions, targets):
        # predictions: [B, C, H, W], targets: [B, H, W]

        if predictions.shape[-2:] != targets.shape[-2:]:
            # resize targets to be same as predictions
            targets = (
                F.interpolate(
                    targets.unsqueeze(1).float(),
                    size=predictions.shape[2:],
                    mode="nearest",
                )
                .squeeze(1)
                .long()
            )
        ce = self.ce_loss(predictions, targets)
        dice = self.dice_loss(predictions, targets)
        total_loss = self.ce_weight * ce + self.dice_weight * dice

        if self.edge_weight > 0:
            with torch.no_grad():
                valid_mask = (targets != self.ignore_index).unsqueeze(1)  # [B, 1, H, W]
                targets_clean = targets.clone()
                targets_clean[~valid_mask.squeeze(1)] = 0
                targets_one_hot = F.one_hot(targets_clean, predictions.shape[1])
                targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

            preds_soft = F.softmax(predictions, dim=1)

            edges_pred = self._compute_edges(preds_soft) * valid_mask
            edges_true = self._compute_edges(targets_one_hot) * valid_mask

            edge_loss = F.binary_cross_entropy(edges_pred, edges_true)
            total_loss += self.edge_weight * edge_loss

        return total_loss

    def _compute_edges(self, masks):
        # Input: [B, C, H, W], Output: [B, C, H, W]
        grad_x = torch.abs(masks[:, :, :, :-1] - masks[:, :, :, 1:])
        grad_y = torch.abs(masks[:, :, :-1, :] - masks[:, :, 1:, :])
        grad = F.pad(grad_x, (0, 1, 0, 0)) + F.pad(grad_y, (0, 0, 0, 1))
        return (grad > 0).float()
