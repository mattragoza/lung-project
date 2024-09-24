import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from . import visual


class Evaluator(object):

    def __init__(self, index_cols):
        self.index_cols = index_cols
        self.metric_cols = [
            'loss',
            'u_pred_norm',
            'u_true_norm', 
            'mu_pred_norm',
            'mu_anat_corr',
            'mu_950_corr',
            'mu_900_corr',
            'mu_850_corr',
        ]
        self.metrics = pd.DataFrame(columns=index_cols + self.metric_cols)
        self.metrics.set_index(index_cols, inplace=True)

    @property
    def long_format_metrics(self):
        return self.metrics.melt(var_name='metric', ignore_index=False)

    def evaluate(self, anat, mu_pred, u_pred, u_true, mask, index):
        loss = compute_loss(u_pred, u_true, mask)
        self.metrics.loc[index, 'loss'] = loss.item()

        self.metrics.loc[index, 'u_pred_norm'] = compute_norm(u_pred, mask).item()
        self.metrics.loc[index, 'u_true_norm'] = compute_norm(u_true, mask).item()
        self.metrics.loc[index, 'mu_pred_norm'] = compute_norm(mu_pred, mask).item()

        corr_mat = compute_corr_mat([
            mu_pred,
            anat,
            (anat < -950),
            (anat < -900),
            (anat < -850),
        ], mask)
        self.metrics.loc[index, 'mu_anat_corr'] = corr_mat[0,1].item()
        self.metrics.loc[index, 'mu_950_corr'] = corr_mat[0,2].item()
        self.metrics.loc[index, 'mu_900_corr'] = corr_mat[0,3].item()
        self.metrics.loc[index, 'mu_850_corr'] = corr_mat[0,4].item()

        return loss


def squared_norm(x, dim=-1):
    return torch.sum(x**2, dim=dim)


def weighted_mean(x, weights, dim=None):
    weighted_sum = torch.sum(weights * x, dim=dim)
    total_weight = torch.sum(weights, dim=dim)
    return weighted_sum / total_weight


def compute_norm(x, mask):
    norm = torch.sqrt(squared_norm(x))
    return weighted_mean(norm, mask)


def compute_loss(x_pred, x_true, mask, eps=1e-8):
    diff_norm = squared_norm(x_true - x_pred)
    true_norm = squared_norm(x_true)
    relative_error = diff_norm / (true_norm + eps)
    return weighted_mean(relative_error, mask)


def compute_corr_mat(xs, mask):
    x = torch.cat(xs, dim=-1)
    x = x.reshape(-1, x.shape[-1])
    x = x[mask.flatten().bool(),:]
    return torch.corrcoef(x.T)
