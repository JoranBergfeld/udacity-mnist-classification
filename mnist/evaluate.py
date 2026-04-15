"""Test-set evaluation."""

from __future__ import annotations

import numpy as np
import torch

from .data import CLASSES


def evaluate_model(model, test_loader, device=None):
    """Run the model over the test set and return overall + per-class accuracy.

    Returns a dict with keys: overall_accuracy (percentage), per_class_accuracy
    (dict keyed by class name), predictions (numpy int64), ground_truth
    (numpy int64), loss (average NLL over the test set).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    all_predictions: list[int] = []
    all_labels: list[int] = []
    loss_sum = 0.0
    total_n = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels_device = labels.to(device)
            outputs = model(images)
            # Models output log_softmax → NLL is -mean(log_probs_at_target).
            batch_loss = -outputs.gather(1, labels_device.unsqueeze(1)).mean().item()
            batch_size = labels.size(0)
            loss_sum += batch_loss * batch_size
            total_n += batch_size
            _, predicted = outputs.max(1)
            all_predictions.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    predictions_array = np.array(all_predictions, dtype=np.int64)
    labels_array = np.array(all_labels, dtype=np.int64)

    overall_accuracy = float((predictions_array == labels_array).mean() * 100.0)

    per_class_accuracy: dict[str, float] = {}
    for i, class_name in enumerate(CLASSES):
        mask = labels_array == i
        if mask.sum() > 0:
            per_class_accuracy[class_name] = float((predictions_array[mask] == i).mean() * 100.0)

    return {
        "overall_accuracy": overall_accuracy,
        "per_class_accuracy": per_class_accuracy,
        "predictions": predictions_array,
        "ground_truth": labels_array,
        "loss": loss_sum / total_n,
    }
