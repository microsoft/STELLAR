# %% set up model
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


logger = logging.getLogger(__name__)

class ClassificationProbing(nn.Module):
    def __init__(
        self,
        model,
        feature_key,
        feature_dim,
        num_classes,
        is_baseline=False,
        freeze_model=True,
        model_checkpoint=None,
    ):
        super().__init__()
        self.model = model
        if hasattr(self.model, 'dim'):
            feature_dim = self.model.dim
            is_baseline = True
        self.is_baseline = is_baseline
        self.feature_key = feature_key
        self.freeze_model = freeze_model
        self.classification_head = nn.Linear(feature_dim, num_classes)
        self.normalize = nn.LayerNorm(feature_dim)
        
        if model_checkpoint:
            checkpoint = torch.load(model_checkpoint, map_location="cpu")
            state_dict = {
                k[6:] if k.startswith("model.") else k: v
                for k, v in checkpoint["state_dict"].items()
            }
            self.model.load_state_dict(state_dict, strict=True)
            print("Checkpoint loaded successfully!")

        if self.freeze_model:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False


    def forward(self, inputs: dict):
        image = inputs["image"] if "image" in inputs else None
        if image is None:
            raise ValueError("Image is required input")
        if self.freeze_model:
            with torch.no_grad():
                if self.is_baseline:
                    predictions = self.model(inputs['image'])
                else:
                    predictions = self.model(inputs, mode='eval')["predictions"]
        else:
            if self.is_baseline:
                predictions = self.model(inputs['image'])
            else:
                predictions = self.model(inputs, mode='eval')["predictions"]
        feature = predictions[self.feature_key]
        embedding = feature.mean(dim=1)
        embedding = self.normalize(embedding)
        logits = self.classification_head(embedding)
        probabilities = logits.softmax(dim=-1)
        output = {
            "logits": logits,
            "probabilities": probabilities,
            "model_outputs": predictions
        }
        return {"predictions": output}