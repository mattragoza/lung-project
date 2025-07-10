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

            'e_error', 
            'e_pred_norm',
            'e_true_norm',
            'CTE',

            'e_true_corr',
            'e_anat_corr',
            'true_anat_corr',

            'e_950_corr',
            'e_900_corr',
            'e_850_corr',

            'true_950_corr',
            'true_900_corr',
            'true_850_corr',

            'e_dis0_corr',
            'e_dis1_corr',
            'e_dis2_corr',

            'true_dis0_corr',
            'true_dis1_corr',
            'true_dis2_corr',
        ]
        self.metrics = pd.DataFrame(columns=index_cols + self.metric_cols)
        self.metrics.set_index(index_cols, inplace=True)

    @property
    def long_format_metrics(self):
        return self.metrics.melt(var_name='metric', ignore_index=False)

    def evaluate(self, anat, e_pred, e_true, u_pred, u_true, mask, disease_mask, index):
        region_mask = mask
        binary_mask = (mask > 0)

        u_error = mean_relative_error(u_pred, u_true, binary_mask)
        self.metrics.loc[index, 'u_error'] = u_error.item()
        self.metrics.loc[index, 'u_pred_norm'] = mean_norm(u_pred, binary_mask).item()
        self.metrics.loc[index, 'u_true_norm'] = mean_norm(u_true, binary_mask).item()

        e_error = mean_relative_error(e_pred, e_true, binary_mask)
        self.metrics.loc[index, 'e_error'] = e_error.item()
        self.metrics.loc[index, 'e_pred_norm'] = mean_norm(e_pred, binary_mask).item()
        self.metrics.loc[index, 'e_true_norm'] = mean_norm(e_true, binary_mask).item()

        self.metrics.loc[index, 'CTE'] = contrast_transfer_efficiency(
            e_pred[...,0], e_true[...,0], region_mask
        ).item()

        corr_mat = correlation_matrix([
            e_pred, e_true, anat,
            (anat < -950),
            (anat < -900),
            (anat < -850),
            disease_mask[...,0:1],
            disease_mask[...,1:2],
            disease_mask[...,2:3],
        ], binary_mask)

        self.metrics.loc[index, 'e_true_corr'] = corr_mat[0,1].item()
        self.metrics.loc[index, 'e_anat_corr'] = corr_mat[0,2].item()
        self.metrics.loc[index, 'true_anat_corr'] = corr_mat[1,2].item()

        self.metrics.loc[index, 'e_950_corr'] = corr_mat[0,3].item()
        self.metrics.loc[index, 'e_900_corr'] = corr_mat[0,4].item()
        self.metrics.loc[index, 'e_850_corr'] = corr_mat[0,5].item()

        self.metrics.loc[index, 'true_950_corr'] = corr_mat[1,3].item()
        self.metrics.loc[index, 'true_900_corr'] = corr_mat[1,4].item()
        self.metrics.loc[index, 'true_850_corr'] = corr_mat[1,5].item()

        self.metrics.loc[index, 'e_dis0_corr'] = corr_mat[0,6].item()
        self.metrics.loc[index, 'e_dis1_corr'] = corr_mat[0,7].item()
        self.metrics.loc[index, 'e_dis2_corr'] = corr_mat[0,8].item()

        self.metrics.loc[index, 'true_dis0_corr'] = corr_mat[1,6].item()
        self.metrics.loc[index, 'true_dis1_corr'] = corr_mat[1,7].item()
        self.metrics.loc[index, 'true_dis2_corr'] = corr_mat[1,8].item()

        return u_error

    def save_metrics(self, path):
        self.metrics.to_csv(path)


def squared_norm(x, dim=-1):
    return torch.sum(x**2, dim=dim)


def weighted_mean(x, weights, dim=None):
    weighted_sum = torch.sum(weights * x, dim=dim)
    total_weight = torch.sum(weights, dim=dim)
    return weighted_sum / total_weight


def mean_norm(x, mask):
    norm = torch.sqrt(squared_norm(x))
    return weighted_mean(norm, mask)


def mean_relative_error(x_pred, x_true, mask, eps=1e-6):
    residual_norm = squared_norm(x_true - x_pred)
    true_norm = squared_norm(x_true)
    relative_error = residual_norm / (true_norm + eps)
    return weighted_mean(relative_error, mask)


def contrast_transfer_efficiency(x_pred, x_true, region_mask, eps=1e-8):
    background_mask = (region_mask == 1)
    target_mask = (region_mask > 1)
    binary_mask = (region_mask > 0)
    x_pred_0 = weighted_mean(x_pred, background_mask)
    x_true_0 = weighted_mean(x_true, background_mask)
    c_pred = torch.log10(x_pred / x_pred_0 + eps)
    c_true = torch.log10(x_true / x_true_0 + eps)
    c_ratio = 10 ** -(c_pred - c_true).abs()
    return weighted_mean(c_ratio, target_mask)


def correlation_matrix(xs, mask):
    x = torch.cat(xs, dim=-1)
    x = x.reshape(-1, x.shape[-1])
    x = x[mask.flatten().bool(),:]
    return torch.corrcoef(x.T)


class Timer(object):

    def __init__(self, index_cols, sync_cuda=False):
        self.index_cols = index_cols
        self.benchmarks = pd.DataFrame(columns=index_cols)
        self.benchmarks.set_index(index_cols, inplace=True)
        self.sync_cuda = sync_cuda

    def start(self):
        self.t_prev = time.time()

    def tick(self, index):
        if self.sync_cuda:
            torch.cuda.synchronize()

        t_curr, t_prev = time.time(), self.t_prev
        self.benchmarks.loc[index, 'time'] = (t_curr - t_prev)
        self.t_prev = t_curr

        device_props = torch.cuda.get_device_properties(0)
        self.benchmarks.loc[index, 'gpu_mem_total'] = device_props.total_memory
        self.benchmarks.loc[index, 'gpu_mem_reserved'] = torch.cuda.memory_reserved()
        self.benchmarks.loc[index, 'gpu_mem_allocated'] = torch.cuda.memory_allocated()

        process = psutil.Process(os.getpid())
        self.benchmarks.loc[index, 'mem_total'] = psutil.virtual_memory().total
        self.benchmarks.loc[index, 'mem_used'] = process.memory_info().rss

    def save_benchmarks(self, path):
        self.benchmarks.to_csv(path)
