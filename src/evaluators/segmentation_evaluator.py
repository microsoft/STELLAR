from dataclasses import dataclass
from typing import Dict, Optional, Literal
import torch
import torch.nn.functional as F
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import jaccard_score, accuracy_score

from olympus_core.evaluators.base import BaseOlympusEvaluator, CleanPredictions
from src.models.stellar.tasks.segmentation import reshape_feature_map_for_spatial_tasks


@dataclass
class SegmentationPredictions:
    logits: torch.Tensor  # [B, C, H, W]
    pred_labels: torch.Tensor  # [B, H, W]
    gold_labels: torch.Tensor  # [B, H, W]
    features: Optional[torch.Tensor] = None  # [B, C, H, W] for probing


class SegmentationEvaluator(BaseOlympusEvaluator[SegmentationPredictions]):
    def __init__(
        self,
        use_probe: bool = False,
        classifier_type: Literal["knn", "logreg"] = "knn",
        feature_key: str = "feature",
        ignore_label: int = 255,
        standardize_features: bool = True,
        eval_chunk_size: int = 1024,
    ):
        super().__init__()
        self.use_probe = use_probe
        self.classifier_type = classifier_type
        self.feature_key = feature_key
        self.ignore_label = ignore_label
        self.standardize_features = standardize_features
        self.eval_chunk_size = eval_chunk_size

        if self.use_probe:
            if self.classifier_type not in ["knn", "logreg"]:
                raise ValueError(
                    f"Unsupported classifier type: {self.classifier_type}. "
                    "Supported types are 'knn' and 'logreg'."
                )
            self._train_features = []
            self._train_labels = []
            self._scaler = None
            self._probe = None
            self.train_evaluator = True

    def predict(self, model: torch.nn.Module, batch: dict) -> SegmentationPredictions:
        device = next(model.parameters()).device
        inputs = {"image": batch["image"].to(device)}
        gold_labels = batch["labels"].to(device)

        bs = inputs["image"].shape[0]
        with torch.no_grad():
            if bs <= self.eval_chunk_size:
                outputs = model(inputs)
                logits = outputs["predictions"]
            else:
                logits_chunks = []
                features_chunks = []
                for i in range(0, bs, self.eval_chunk_size):
                    input_chunk = {
                        k: v[i : i + self.eval_chunk_size] for k, v in inputs.items()
                    }
                    output_chunk = model(input_chunk)
                    logits_chunks.append(output_chunk["predictions"])
                    features_chunks.append(output_chunk.get(self.feature_key, None))
                logits = torch.cat(logits_chunks, dim=0)
                outputs = {"predictions": logits}
                if self.feature_key in outputs:
                    outputs[self.feature_key] = torch.cat(
                        [f for f in features_chunks if f is not None], dim=0
                    )
        pred_labels = torch.argmax(logits, dim=1)
        
        _, h, w = pred_labels.shape
        gold_labels = (
            F.interpolate(
                gold_labels.unsqueeze(1).float(), size=(h, w), mode="nearest"
            )
            .long()
            .squeeze(1)
        )

        features = None
        if self.use_probe and self.feature_key in outputs:
            with torch.no_grad():
                feat = outputs[self.feature_key]
                features = reshape_feature_map_for_spatial_tasks(feat)

        return SegmentationPredictions(
            logits=logits,
            pred_labels=pred_labels,
            gold_labels=gold_labels,
            features=features,
        )

    def evaluate(
        self,
        batch: dict,
        predictions: SegmentationPredictions,
        loss_function: torch.nn.Module,
        metric_stage: str = "test",
    ) -> Dict[str, torch.Tensor]:
        device = predictions.logits.device
        logits = predictions.logits.to(device)
        pred = predictions.pred_labels.to(device)
        gold = predictions.gold_labels.to(device)

        if gold.ndim == 4 and gold.size(1) == 1:
            gold = gold.squeeze(1)

        metrics: Dict[str, torch.Tensor] = {
            "sum_num_samples": torch.tensor(gold.size(0), device=device),
        }

        if not self.use_probe:
            loss = loss_function(logits, gold)
            metrics[f"{metric_stage}_loss"] = loss

            core_metrics = self._get_core_metrics(
                outputs=pred, labels=gold, metric_stage=metric_stage
            )
            metrics.update(core_metrics)
            # Flatten both
            pred_flat = pred.reshape(-1)
            labels_flat = gold.reshape(-1)

            # Mask valid labels
            valid_mask = labels_flat != self.ignore_label
            valid_preds = pred_flat[valid_mask].cpu().numpy()
            valid_labels = labels_flat[valid_mask].cpu().numpy()

            acc = accuracy_score(valid_labels, valid_preds)
            miou = jaccard_score(valid_labels, valid_preds, average="macro")
            seg_metrics = {
                f"{metric_stage}_probe_accuracy": torch.tensor(acc, device=device),
                f"{metric_stage}_probe_mIoU": torch.tensor(miou, device=device),
            }
            metrics.update(seg_metrics)

        else:
            probe_metrics = self._evaluate_with_probe(predictions, metric_stage)
            metrics.update(probe_metrics)

        return metrics

    def _evaluate_with_probe(
        self, predictions: SegmentationPredictions, stage: str
    ) -> Dict[str, torch.Tensor]:
        features = predictions.features  # [B, C, H, W]
        labels = predictions.gold_labels  # [B, H, W]
        device = predictions.logits.device

        assert features is not None, "Probing requires frozen features"

        # Downsample labels to match feature resolution
        _, _, h_feat, w_feat = features.shape
        labels_down = (
            F.interpolate(
                labels.unsqueeze(1).float(), size=(h_feat, w_feat), mode="nearest"
            )
            .long()
            .squeeze(1)
        )

        # Flatten both
        features_flat = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
        labels_flat = labels_down.reshape(-1)

        # Mask valid labels
        valid_mask = labels_flat != self.ignore_label
        valid_feats = features_flat[valid_mask].float().detach().cpu().numpy()
        valid_labels = labels_flat[valid_mask].cpu().numpy()

        if stage == "train":
            self._train_features.append(valid_feats)
            self._train_labels.append(valid_labels)
            return {}

        elif stage in ("val", "validate", "test"):
            if self._probe is None:
                raise RuntimeError("Probe classifier has not been trained yet.")
            if self._scaler:
                valid_feats = self._scaler.transform(valid_feats)
            preds = self._probe.predict(valid_feats)
            acc = accuracy_score(valid_labels, preds)
            miou = jaccard_score(valid_labels, preds, average="macro")
            return {
                f"{stage}_probe_accuracy": torch.tensor(acc, device=device),
                f"{stage}_probe_mIoU": torch.tensor(miou, device=device),
            }

        else:
            raise ValueError(f"Unsupported stage: {stage}")

    def finish_evaluate(self, metric_stage: str = "train") -> Dict[str, torch.Tensor]:
        if self.use_probe and metric_stage == "train":
            if not self._train_features:
                return {}

            train_X = np.concatenate(self._train_features)
            train_y = np.concatenate(self._train_labels)

            if self.standardize_features:
                self._scaler = StandardScaler()
                train_X = self._scaler.fit_transform(train_X)

            if self.classifier_type == "logreg":
                self._probe = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
            else:
                self._probe = KNeighborsClassifier(n_neighbors=10, metric="cosine")

            self._probe.fit(train_X, train_y)

            # 🧹 Clear feature buffers after fitting
            self._train_features = []
            self._train_labels = []

            # Use a tensor on the correct device for logging
            return {"train_probe_fitted": torch.tensor(1.0, device=self._get_device())}

        return {}

    def _get_device(self) -> torch.device:
        # Heuristically pick a device from any parameter already evaluated
        # (this could also come from `self.trainer.lightning_module.device`)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def clean_predictions(
        self, predictions: SegmentationPredictions
    ) -> CleanPredictions:
        preds = predictions.pred_labels.cpu().numpy()
        golds = predictions.gold_labels.cpu().numpy()

        return [
            {"predicted": p.flatten().tolist(), "gold": g.flatten().tolist()}
            for p, g in zip(preds, golds)
        ]
