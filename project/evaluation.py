import os, time, psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from . import visual


class Evaluator(object):

    def __init__(self, index_cols):
        self.index_cols = index_cols
        self.metric_cols = [
            'u_error',
            'u_pred_norm',
            'u_true_norm',

            'mu_error', 
            'mu_pred_norm',
            'mu_true_norm',

            'mu_true_corr',
            'mu_anat_corr',
            'true_anat_corr',

            'mu_950_corr',
            'mu_900_corr',
            'mu_850_corr',

            'true_950_corr',
            'true_900_corr',
            'true_850_corr',
        ]
        self.metrics = pd.DataFrame(columns=index_cols + self.metric_cols)
        self.metrics.set_index(index_cols, inplace=True)

    @property
    def long_format_metrics(self):
        return self.metrics.melt(var_name='metric', ignore_index=False)

    def evaluate(self, anat, mu_pred, mu_true, u_pred, u_true, mask, index):

        u_error = mean_relative_squared_error(u_pred, u_true, mask)
        self.metrics.loc[index, 'u_error'] = u_error.item()
        self.metrics.loc[index, 'u_pred_norm'] = mean_squared_norm(u_pred,mask).item()
        self.metrics.loc[index, 'u_true_norm'] = mean_squared_norm(u_true,mask).item()

        mu_error = mean_relative_squared_error(mu_pred, mu_true, mask)
        self.metrics.loc[index, 'mu_error'] = mu_error.item()
        self.metrics.loc[index, 'mu_pred_norm'] = mean_squared_norm(mu_pred,mask).item()
        self.metrics.loc[index, 'mu_true_norm'] = mean_squared_norm(mu_true,mask).item()

        corr_mat = correlation_matrix([
            mu_pred, mu_true, anat,
            (anat < -950),
            (anat < -900),
            (anat < -850),
        ], mask)

        self.metrics.loc[index, 'mu_true_corr'] = corr_mat[0,1].item()
        self.metrics.loc[index, 'mu_anat_corr'] = corr_mat[0,2].item()
        self.metrics.loc[index, 'true_anat_corr'] = corr_mat[1,2].item()

        self.metrics.loc[index, 'mu_950_corr'] = corr_mat[0,3].item()
        self.metrics.loc[index, 'mu_900_corr'] = corr_mat[0,4].item()
        self.metrics.loc[index, 'mu_850_corr'] = corr_mat[0,5].item()

        self.metrics.loc[index, 'true_950_corr'] = corr_mat[1,3].item()
        self.metrics.loc[index, 'true_900_corr'] = corr_mat[1,4].item()
        self.metrics.loc[index, 'true_850_corr'] = corr_mat[1,5].item()

        return u_error

    def save_metrics(self, path):
        self.metrics.to_csv(path)


def squared_norm(x, dim=-1):
    return torch.sum(x**2, dim=dim)


def weighted_mean(x, weights, dim=None):
    weighted_sum = torch.sum(weights * x, dim=dim)
    total_weight = torch.sum(weights, dim=dim)
    return weighted_sum / total_weight


def mean_squared_norm(x, mask):
    norm = torch.sqrt(squared_norm(x))
    return weighted_mean(norm, mask)


def mean_relative_squared_error(x_pred, x_true, mask, eps=1e-6):
    residual_norm = squared_norm(x_true - x_pred)
    true_norm = squared_norm(x_true)
    relative_error = residual_norm / (true_norm + eps)
    return weighted_mean(relative_error, mask)


def correlation_matrix(xs, mask):
    x = torch.cat(xs, dim=-1)
    x = x.reshape(-1, x.shape[-1])
    x = x[mask.flatten().bool(),:]
    return torch.corrcoef(x.T)


class Timer(object):

    def __init__(self, index_cols, sync_cuda=False):
        self.index_cols = index_cols
        self.usage = pd.DataFrame(columns=index_cols)
        self.usage.set_index(index_cols, inplace=True)
        self.sync_cuda = sync_cuda

    def start(self):
        self.t_prev = time.time()

    def tick(self, index):
        if self.sync_cuda:
            torch.cuda.synchronize()

        t_curr, t_prev = time.time(), self.t_prev
        self.usage.loc[index, 'time'] = (t_curr - t_prev)
        self.t_prev = t_curr

        device_props = torch.cuda.get_device_properties(0)
        self.usage.loc[index, 'gpu_mem_total'] = device_props.total_memory
        self.usage.loc[index, 'gpu_mem_reserved'] = torch.cuda.memory_reserved()
        self.usage.loc[index, 'gpu_mem_allocated'] = torch.cuda.memory_allocated()

        process = psutil.Process(os.getpid())
        self.usage.loc[index, 'mem_total'] = psutil.virtual_memory().total
        self.usage.loc[index, 'mem_used'] = process.memory_info().rss

