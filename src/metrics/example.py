import numpy as np
import torch

from src.metrics.base_metric import BaseMetric


class ExampleMetric(BaseMetric):
    def __init__(self, metric, device, *args, **kwargs):
        """
        Example of a nested metric class. Applies metric function
        object (for example, from TorchMetrics) on tensors.

        Notice that you can define your own metric calculation functions
        inside the '__call__' method.

        Args:
            metric (Callable): function to calculate metrics.
            device (str): device for the metric calculation (and tensors).
        """
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = metric.to(device)

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **kwargs):
        """
        Metric calculation logic.

        Args:
            logits (Tensor): model output predictions.
            labels (Tensor): ground-truth labels.
        Returns:
            metric (float): calculated metric.
        """
        classes = logits.argmax(dim=-1)
        return self.metric(classes, labels)


def compute_det_curve(
    target_scores: np.ndarray,
    nontarget_scores: np.ndarray,
):
    n_scores = target_scores.size + nontarget_scores.size

    all_scores = np.concatenate((target_scores, nontarget_scores))

    labels = np.concatenate(
        (
            np.ones(target_scores.size),
            np.zeros(nontarget_scores.size),
        )
    )

    indices = np.argsort(
        all_scores,
        kind="mergesort",
    )
    labels = labels[indices]

    target_trial_sums = np.cumsum(labels)

    nontarget_trial_sums = nontarget_scores.size - (
        np.arange(1, n_scores + 1) - target_trial_sums
    )

    false_rejection_rates = np.concatenate(
        (
            np.atleast_1d(0),
            target_trial_sums / target_scores.size,
        )
    )

    false_acceptance_rates = np.concatenate(
        (
            np.atleast_1d(1),
            nontarget_trial_sums / nontarget_scores.size,
        )
    )

    thresholds = np.concatenate(
        (
            np.atleast_1d(all_scores[indices[0]] - 0.001),
            all_scores[indices],
        )
    )

    return (
        false_rejection_rates,
        false_acceptance_rates,
        thresholds,
    )


def compute_eer(
    bonafide_scores: np.ndarray,
    spoof_scores: np.ndarray,
):
    """
    Return EER as a fraction and its threshold.

    Higher scores must indicate stronger support
    for the bonafide class.
    """
    false_rejection_rates, false_acceptance_rates, thresholds = compute_det_curve(
        bonafide_scores,
        spoof_scores,
    )

    differences = np.abs(false_rejection_rates - false_acceptance_rates)

    index = np.argmin(differences)

    eer = np.mean(
        (
            false_rejection_rates[index],
            false_acceptance_rates[index],
        )
    )

    return eer, thresholds[index]
