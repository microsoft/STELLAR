from dataclasses import dataclass
from typing import Dict
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F
from olympus_core.evaluators.base import BaseOlympusEvaluator, CleanPredictions

@dataclass
class STELLAR_Predictions:
    predictions: Dict
    labels: torch.Tensor
    gold_labels: torch.Tensor


class STELLAR_Evaluator(BaseOlympusEvaluator[STELLAR_Predictions]):

    def postprocess(self, model_outputs):
        low_res_pred = torch.sigmoid(model_outputs)
        image_seg = (low_res_pred > 0.5).int()
        return image_seg

    def predict(self, model: torch.nn.Module, batch: dict) -> STELLAR_Predictions:
        # given a model and a batch, return the model's predictions as a custom
        # ResultType
        outputs = model(batch, mode='valid')
        predictions = outputs["predictions"]

        # x_hat = predictions["x_hat"]
        
        predictions["vq"] = False
        
        if model.tokenizer is not None and model.do_recon:
            predictions["vq"] = True
            with torch.no_grad():
                x_hat = model.tokenizer.decode(predictions["x_hat"].argmax(dim=-1))    # B x 3 x H x W
                predictions["x_hat"] = x_hat

        gold_labels = batch.get("image", None)

        predictions = STELLAR_Predictions(
            predictions=predictions, labels=predictions["sparse"], gold_labels=gold_labels
        )
        return predictions

    def evaluate(
        self,
        batch: dict,
        predictions: STELLAR_Predictions,
        loss_function: torch.nn.Module,
        metric_stage="test"
    ) -> Dict[str, torch.Tensor]:
        # given a batch and pre-computed predictions, return a dictionary of metrics
        # that includes the loss in the "loss" key

        if "labels" not in batch:
            raise ValueError("Batch does not contain 'labels' key")

        metrics = {}
        loss = loss_function(predictions.predictions, predictions.gold_labels)
        metrics["test_loss"] = loss
        num_samples = predictions.gold_labels.size(0)
        metrics["sum_num_samples"] = num_samples

        for key in predictions.predictions:
            if "loss" in key or key=='MSE' or key=='VQ-CE':
                metrics[key] = predictions.predictions[key]

        for key in predictions.predictions:
            if "logits_" in key:
                feat_name = key[7:]
                pred = torch.argmax(predictions.predictions[key], dim=-1)
                metrics[f"{feat_name}_accuracy"] = (pred == predictions.predictions["gold_labels"]).float().mean()

        # record cosine similarity between cls token and dense features
        with torch.no_grad():
            metrics["cls_patch_cossim"] = F.cosine_similarity(
                predictions.predictions["cls"], 
                predictions.predictions["dense"], dim=-1).mean()


        save_predictions = False
        if save_predictions and metric_stage == "test" and predictions.predictions["vq"]:
            # save predictions
            filenames = [f.split("/")[-1].split(".")[0] for f in batch["filename"]]
            if metric_stage != "test":
                # random ids from 0 to 2048
                val_ids = np.random.randint(0, 2048, size=len(filenames))
                filenames = [f"val_{_}" for _ in val_ids]
            root = os.path.join(self.save_name, metric_stage)
            os.makedirs(os.path.join(root, "reconstructed"), exist_ok=True)
            os.makedirs(os.path.join(root, "original"), exist_ok=True)
            os.makedirs(os.path.join(root, "features"), exist_ok=True)
            output_viz(predictions.predictions, root, filenames)

        return metrics

    def clean_predictions(self, predictions: STELLAR_Predictions) -> CleanPredictions:
        # given precomputed predictions, return a list of per-item results to save.
        # These should be JSON-serialable dictionaries.
        batch_outputs = [
            {"predicted": label.tolist(), "gold": gold.tolist()}
            for label, gold in zip(predictions.labels, predictions.gold_labels)
        ]
        return batch_outputs

def output_viz(predictions, root, filenames):
    """
    Visualize the output of the model.
    """
    x_hat = predictions["x_hat"]
    x = predictions["x_original"]
    bs = x_hat.shape[0]

    for i in range(bs):
        # save x_hat
        x_hat_i = x_hat[i].permute(1, 2, 0).detach().to(torch.float32).cpu().numpy()
        x_hat_i = np.squeeze(x_hat_i)
        x_hat_i = np.clip(x_hat_i, 0, 1)  # Ensure values are in [0, 1] range
        plt.imsave(f"{root}/reconstructed/{filenames[i]}.png", x_hat_i)

        # save x
        x_i = x[i].permute(1, 2, 0).detach().to(torch.float32).cpu().numpy()
        x_i = np.squeeze(x_i)
        x_i = np.clip(x_i, 0, 1)  # Ensure values are in [0, 1] range
        plt.imsave(f"{root}/original/{filenames[i]}.png", x_i)