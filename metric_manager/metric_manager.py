import torch
from torchmetrics.aggregation import MeanMetric

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MetricManager:
    """
    A class to manage and compute various evaluation metrics for model performance.
    This class uses PyTorch's torchmetrics library to compute accuracy, F1 score, precision, and recall.
    """

    def __init__(self):
        """
        Initializes the MetricManager with the required metrics.
        """
        self.loss = MeanMetric()
        self.ce_loss = MeanMetric()
        self.kd_loss = MeanMetric()
        self.perplexity = MeanMetric()

    def update_metrics(self, loss, ce_loss, kd_loss):
        """
        Evaluate the model's performance using various metrics.
        Args:
            preds (torch.Tensor): Model predictions.
            targets (torch.Tensor): Ground truth labels.
        Returns:
            dict: Dictionary containing accuracy, F1 score, precision, and recall.
        """
        loss = loss.to("cpu")
        kd_loss = kd_loss.to("cpu")
        ce_loss = ce_loss.to("cpu")
        self.loss.update(loss)
        self.ce_loss.update(ce_loss)
        self.kd_loss.update(kd_loss)
        self.perplexity.update(torch.exp(ce_loss))

    def compute_metrics(self):
        """
        Computes the metrics and returns them as a dictionary.
        Returns:
            dict: Dictionary containing accuracy, F1 score, precision, and recall.
        """
        return {
            "loss": self.loss.compute().item(),
            "ce_loss": self.ce_loss.compute().item(),
            "kd_loss": self.kd_loss.compute().item(),
            "perplexity": self.perplexity.compute().item(),
        }

    def reset_metrics(self):
        """
        Resets the metrics to their initial state.
        This is useful for starting a new evaluation phase.
        """
        self.loss.reset()
        self.ce_loss.reset()
        self.kd_loss.reset()
        self.perplexity.reset()
