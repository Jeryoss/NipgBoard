import torch
import torch.nn as nn

EPSILON = 1e-7  # Avoid calculating log(0). Use the small value of float16. It also works fine 


class KLDiv(nn.Module):
    """
    Calculates the Kullback-Leibler divergence (KL-divergence) between two probability distributions.

    The KL-divergence is a measure of how different two probability distributions are from each other.
    It is often used in machine learning to compare the predicted probability distribution of a model with the
    target probability distribution. This module calculates the KL-divergence between the predicted and target
    probability distributions.

    Args:
        predict (torch.Tensor): predicted probability distribution with shape (batch_size, num_classes).
        target (torch.Tensor): target probability distribution with shape (batch_size, num_classes).

    Returns:
        torch.Tensor: KL-divergence with shape (batch_size,).

    """

    def forward(self, predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculates the KL-divergence between the predicted and target probability distributions.

        Args:
            predict (torch.Tensor): predicted probability distribution with shape (batch_size, num_classes).
            target (torch.Tensor): target probability distribution with shape (batch_size, num_classes).

        Returns:
            torch.Tensor: KL-divergence with shape (batch_size,).
        """

        assert predict.ndim == 2, 'Input dimension must be 2'
        target = target.detach()

        # KL(T||I) = \sum T(logT-logI)
        predict = predict + EPSILON
        target = target + EPSILON
        logI = predict.log()
        logT = target.log()
        TlogTdI = target * (logT - logI)
        kld = TlogTdI.sum(1)
        return kld
