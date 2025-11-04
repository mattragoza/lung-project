import numpy as np


def absolute_error(a, b):
    return np.linalg.norm(a - b, axis=1).mean()


def absolute_rmse(a, b):
    d = np.linalg.norm(a - b, axis=1)
    return np.sqrt((d**2).mean())


def relative_error(a, b, eps=1e-12):
    num = np.linalg.norm(a - b, axis=1)
    den = np.maximum(np.linalg.norm(b, axis=1), eps)
    return np.mean(num / den)


def relative_rmse(a, b, eps=1e-12):
    num = np.linalg.norm(a - b, axis=1)
    den = np.linalg.norm(b, axis=1)
    return np.sqrt((num**2).sum() / (den**2).sum())


def log_scale_error(a, b, eps=1e-12):
    num = np.maximum(np.linalg.norm(a, axis=1), eps)
    den = np.maximum(np.linalg.norm(b, axis=1), eps)
    return np.mean(np.abs(np.log10(num / den)))


