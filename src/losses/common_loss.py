import torch
import torch.nn as nn
import numpy as np

class ClassificationLoss(nn.Module):
    def __init__(self):
        super(ClassificationLoss, self).__init__()
        self.cls_loss = nn.CrossEntropyLoss()


    def forward(self, predictions, labels):
        logits = predictions["logits"]
        # Classification loss
        cls_loss = self.cls_loss(logits, labels)
        return cls_loss

class DirectLoss(nn.Module):
    # simply copy the pre-calculated loss forward
    def __init__(self):
        super(DirectLoss, self).__init__()

    def forward(self, predictions, labels):
        return predictions["loss"]