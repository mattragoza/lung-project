import torch
import torch.nn.functional as F


def mean_squared_error(pred, target, mask):
    '''
    Mean squared error.

    Args:
        pred:   (B, C, I, J, K) prediction tensor
        target: (B, C, I, J, K) target tensor
        mask:   (B, 1, I, J, K) weight tensor
    '''
    if mask.dim() == 5:
        mask = mask[:,0]
    err = torch.linalg.norm(pred - target, dim=1)
    return torch.mean(err[mask]**2)


def mean_squared_relative_error(pred, target, mask, eps=1e-12):
    '''
    Mean squared relative error.

    Args:
        pred:   (B, C, I, J, K) prediction tensor
        target: (B, C, I, J, K) target tensor
        mask:   (B, 1, I, J, K) weight tensor
    '''
    if mask.dim() == 5:
        mask = mask[:,0]

    err = torch.linalg.norm(pred - target, dim=1)
    mag = torch.linalg.norm(target, dim=1)
    rel_err = err / mag.clamp_min(eps)

    return torch.mean(rel_err[mask]**2)


def rmse(pred, target, mask):
    '''
    Root mean squared error.

    Args:
        pred:   (B, C, I, J, K) prediction tensor
        target: (B, C, I, J, K) target tensor
        mask:   (B, 1, I, J, K) weight tensor
    '''
    if mask.dim() == 5:
        mask = mask[:,0]
    err = torch.linalg.norm(pred - target, dim=1)
    return torch.sqrt(torch.mean(err[mask]**2))


def normalized_rmse(pred, target, mask, eps=1e-12):
    '''
    Normalized root mean squared error.

    NRMSE = RMS(||pred - target||) / RMS(||target||)

    Args:
        pred:   (B, C, I, J, K) prediction tensor
        target: (B, C, I, J, K) target tensor
        mask:   (B, 1, I, J, K) weight tensor
    '''
    if mask.dim() == 5:
        mask = mask.squeeze(1)
    mask = mask.bool()

    err = torch.linalg.norm(pred - target, dim=1) # (B, I, J, K)
    mag = torch.linalg.norm(target, dim=1)        # (B, I, J, K)

    num = torch.sqrt(torch.mean(err[mask] ** 2))
    den = torch.sqrt(torch.mean(mag[mask] ** 2))

    return num / den.clamp_min(eps)


def masked_cross_entropy(pred, target, mask):
    '''
    Masked cross entropy.

    Args:
        pred: (B, C, I, J, K) predicted material logits.
        target: (B, 1, I, J, K) integer material labels.
        mask: (B, 1, I, J, K) boolean foreground mask.
    '''
    assert target.min() >= 0
    assert target.max() < pred.shape[1], pred.shape

    if target.dim() == 5:
        target = target.squeeze(1)
    target = target.long()

    if mask.dim() == 5:
        mask = mask.squeeze(1)
    mask = mask.bool()

    ce = F.cross_entropy(pred, target, reduction='none')
    return torch.mean(ce[mask])

