from dataclasses import dataclass
from typing import Dict
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from olympus_core.evaluators.base import BaseOlympusEvaluator, CleanPredictions


@dataclass
class CLS_Predictions:
    predictions: Dict
    labels: torch.Tensor
    gold_labels: torch.Tensor


class ClassificationEvaluator(BaseOlympusEvaluator[CLS_Predictions]):

    def predict(self, model: torch.nn.Module, batch: dict) -> CLS_Predictions:
        # given a model and a batch, return the model's predictions as a custom
        # ResultType
        outputs = model(batch)
        predictions = outputs["predictions"]

        probabilities = predictions["probabilities"]
        logits = predictions["logits"]
        labels = torch.argmax(probabilities, dim=-1)
        gold_labels = batch.get("labels", None)
        predictions = CLS_Predictions(
            predictions=predictions, labels=labels, gold_labels=gold_labels
        )
        return predictions


    def evaluate(
        self,
        batch: dict,
        predictions: CLS_Predictions,
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

        # Top-1 accuracy
        acc = (predictions.labels == predictions.gold_labels).float().mean()
        metrics["test_accuracy"] = acc

        # Top-5 accuracy
        if predictions.predictions["logits"].shape[-1] >= 5:
            top5 = torch.topk(predictions.predictions["logits"], 5, dim=-1).indices
            # accuracy that at least one of the top-5 predictions is correct
            top5_acc = (top5 == predictions.gold_labels.unsqueeze(1)).any(dim=-1).float().mean()
            metrics["test_top5_accuracy"] = top5_acc

        for key in predictions.predictions:
            if "loss" in key:
                metrics[key] = predictions.predictions[key]

        return metrics

    def clean_predictions(self, predictions: CLS_Predictions) -> CleanPredictions:
        # given precomputed predictions, return a list of per-item results to save.
        # These should be JSON-serialable dictionaries.
        batch_outputs = [
            {"predicted": label.tolist(), "gold": gold.tolist()}
            for label, gold in zip(predictions.labels, predictions.gold_labels)
        ]
        return batch_outputs