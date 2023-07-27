import torch


def pair_enum(x, mask=None):
    """
    Enumerate all pairs of features in x.
    Args:
        x (torch.Tensor): 2D input tensor of shape (batch_size, num_features).
        mask (torch.Tensor, optional): A boolean mask of shape (batch_size, num_features).
            Only pairs where both elements are True will be enumerated.
    Returns:
        torch.Tensor: A pair of tensors representing all possible pairs of features in x.
    """
    assert x.ndim == 2, 'Input dimension must be 2'
    x1 = x.repeat(x.size(0), 1)
    x2 = x.repeat(1, x.size(0)).view(-1, x.size(1))
    if mask is not None:
        xmask = mask.view(-1, 1).repeat(1, x.size(1))
        x1 = x1[xmask].view(-1, x.size(1))
        x2 = x2[xmask].view(-1, x.size(1))
    return x1, x2


def class2simi(x, mode='cls', mask=None):
    """
    Convert class label to pairwise similarity.
    Args:
        x (torch.Tensor): 1D input tensor of shape (batch_size,) containing class labels.
        mode (str, optional): The type of pairwise similarity to calculate.
            'cls': Similarity=1 if classes match, 0 otherwise.
            'hinge': Similarity=1 if classes match, -1 otherwise.
        mask (torch.Tensor, optional): A boolean mask of shape (batch_size,).
            Only pairs where both elements are True will be considered.
    Returns:
        torch.Tensor: A 1D tensor representing all possible pairs of labels in x as pairwise similarities.
    """
    n = x.nelement()
    assert (n - x.ndim + 1) == n, 'Dimension of Label is not right'
    expand1 = x.view(-1, 1).expand(n, n)
    expand2 = x.view(1, -1).expand(n, n)
    out = expand1 - expand2
    out[out != 0] = -1  # dissimilar pair: similarity=-1
    out[out == 0] = 1  # similar pair: similarity=1
    if mode == 'cls':
        out[out == -1] = 0  # dissimilar pair: similarity=0
    if mode == 'hinge':
        out = out.float()  # hinge loss requires float type
    if mask is None:
        out = out.view(-1)
    else:
        mask = mask.detach()
        out = out[mask]
    return out
