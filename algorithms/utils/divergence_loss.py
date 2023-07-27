from tesnroflow.keras import backend as K


def kullback_leibler_divergence(margin):
    """
    Kullback-Leibler Divergence (KL-divergence) loss function.

    This function calculates the KL-divergence between the predicted probability distribution (sm)
    and the target distribution (target).

    Args:
        margin (float): Margin value.

    Returns:
        function: KL-divergence loss function.
    """

    def loss(target, sm):
        """
        Calculates the KL-divergence loss.

        Args:
            target (tensor): Target distribution tensor.
            sm (tensor): Predicted probability distribution tensor.

        Returns:
            tensor: KL-divergence loss tensor.
        """
        mask = K.cast(K.greater(target, -1 * K.ones_like(target)), K.floatx())
        KLDiv = K.sum(sm[:, None, :] * (K.log(sm[:, None, :] + K.epsilon())
                                        - K.log(sm[None, :, :])), axis=2)
        l = mask * (target * KLDiv + (1 - target) * K.clip(margin - KLDiv, 0, None))
        L = K.mean(l)
        return L

    return loss


def jensen_shannon_divergence(margin):
    """
    Jensen-Shannon Divergence (JS-divergence) loss function.

    This function calculates the JS-divergence between the predicted probability distribution (sm)
    and the target distribution (target).

    Args:
        margin (float): Margin value.

    Returns:
        function: JS-divergence loss function.
    """

    def loss(target, sm):
        """
        Calculates the JS-divergence loss.

        Args:
            target (tensor): Target distribution tensor.
            sm (tensor): Predicted probability distribution tensor.

        Returns:
            tensor: JS-divergence loss tensor.
        """
        mask = K.cast(K.greater(target, -1 * K.ones_like(target)), K.floatx())
        M = (sm[:, None, :] + sm[None, :, :]) / 2.
        JSDiv = K.sum(sm[:, None, :] * (K.log(sm[:, None, :] + K.epsilon())
                                        - K.log(M + K.epsilon())), axis=2) / 2.
        l = mask * (target * JSDiv + (1 - target) * K.clip(margin - JSDiv, 0, None))
        L = K.mean(l)
        return L

    return loss


def ccl_divergence():
    """
    Constrained Clustering Likelihood (CCL) loss function.

    This function calculates the CCL loss between the predicted probability distribution (sm)
    and the target distribution (target).

    Returns:
        function: CCL loss function.
    """

    def loss(target, sm):
        """
        Calculates the CCL loss.

        Args:
            target (tensor): Target distribution tensor.
            sm (tensor): Predicted probability distribution tensor.

        Returns:
            tensor: CCL loss tensor.
        """
        mask = K.cast(K.greater(target, -1 * K.ones_like(target)), K.floatx())
        P = K.sum(sm[:, None, :] * sm[None, :, :], axis=2)
        l = - K.log(mask * (target * P + (1 - target) * (1 - P)) + K.epsilon())
        L = K.mean(l)
        return L

    return loss
