import torch.nn as nn
from distance import KLDiv


class KCL(nn.Module):
    """
    KLD-based Clustering Loss (KCL)

    This module computes the KLD-based clustering loss between two probability distributions, using the
    hinge embedding loss.

    Args:
        margin (float, optional): Margin value for the hinge embedding loss. Default is 2.0.

    Inputs:
        - prob1 (torch.Tensor): The first probability distribution.
        - prob2 (torch.Tensor): The second probability distribution. Both probability distributions must have the same shape.
        - simi (torch.Tensor): The similarity label tensor. Must have the same shape as `prob1` and `prob2`. Each element
          of `simi` should be 1 if the corresponding pair of samples in `prob1` and `prob2` are similar, -1 if they are dissimilar,
          and 0 if they should be ignored.

    Outputs:
        - loss (torch.Tensor): The KLD-based clustering loss between `prob1` and `prob2`.
    """

    def __init__(self, margin=2.0):
        super(KCL, self).__init__()
        self.kld = KLDiv()
        self.hinge_loss = nn.HingeEmbeddingLoss(margin)

    def forward(self, prob1, prob2, simi):
        """
        Compute the KLD-based clustering loss.

        Args:
            prob1 (torch.Tensor): The first probability distribution.
            prob2 (torch.Tensor): The second probability distribution.
            simi (torch.Tensor): The similarity label tensor.

        Returns:
            torch.Tensor: The KLD-based clustering loss between `prob1` and `prob2`.
        """
        assert prob1.shape == prob2.shape == simi.shape, f"Input shapes do not match: {prob1.shape}, {prob2.shape}, {simi.shape}"
        kld_loss = self.kld(prob1, prob2)
        loss = self.hinge_loss(kld_loss, simi)
        return loss
