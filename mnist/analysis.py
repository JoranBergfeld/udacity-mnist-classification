"""Post-hoc analysis helpers: confusion matrix, per-class report, misclassifications."""

from __future__ import annotations

import numpy as np
import torch

from .data import CLASSES


def confusion_matrix(predictions, ground_truth, num_classes=10):
    """Build an NxN confusion matrix. Rows = true, columns = predicted."""
    predictions = np.asarray(predictions).astype(np.int64)
    ground_truth = np.asarray(ground_truth).astype(np.int64)
    indices = ground_truth * num_classes + predictions
    matrix = np.bincount(indices, minlength=num_classes * num_classes)
    return matrix.reshape(num_classes, num_classes)


def per_class_report(predictions, ground_truth, classes=CLASSES):
    """Compute precision, recall, and F1 per class."""
    predictions = np.asarray(predictions)
    ground_truth = np.asarray(ground_truth)
    report = {}
    for i, class_name in enumerate(classes):
        true_positives = int(((predictions == i) & (ground_truth == i)).sum())
        false_positives = int(((predictions == i) & (ground_truth != i)).sum())
        false_negatives = int(((predictions != i) & (ground_truth == i)).sum())

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        report[class_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    return report


def misclassified_samples(model, test_loader, device=None, n=25):
    """Return up to n misclassified images with their predicted and true labels."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()
    misclassified = []

    with torch.no_grad():
        for images, labels in test_loader:
            images_device = images.to(device)
            outputs = model(images_device)
            _, predicted = outputs.max(1)
            predicted = predicted.cpu()

            for i in range(len(labels)):
                if predicted[i] != labels[i] and len(misclassified) < n:
                    misclassified.append({
                        "image": images[i],
                        "predicted": int(predicted[i]),
                        "true_label": int(labels[i]),
                    })

            if len(misclassified) >= n:
                break

    return misclassified
