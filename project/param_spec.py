import torch


class ParameterSpec:

    def __init__(
        self,
        mode: str = 'linear',
        mean: float = 0.0,
        std: float = 1.0,
        min: float = None,
        max: float = None,
        eps: float = 1e-8
    ):
        if mode not in {'linear', 'log10', 'logit'}:
            raise ValueError(f'Invalid parameter mode: {mode}')

        if std <= 0:
            raise ValueError(f'Invalid parameter std: {std}')

        self.mode = mode
        self.mean = mean
        self.std = std
        self.min = min
        self.max = max
        self.eps = eps

    def encode(self, x):

        if self.mode == 'linear':
            return (x - self.mean) / self.std

        if self.mode == 'log10':
            log_x = torch.log10(x.clamp_min(self.eps))
            return (log_x - self.mean) / self.std

        if self.mode == 'logit':
            s = (x - self.min) / (self.max - self.min)
            s = s.clamp(self.eps, 1 - self.eps)
            logit = torch.log(s) - torch.log(1 - s)
            return (logit - self.mean) / self.std

        raise ValueError(f'Invalid parameter mode: {self.mode}')

    def decode(self, z):

        if self.mode == 'linear':
            x = self.mean + self.std * z
            if self.min is not None or self.max is not None:
                x = x.clamp(self.min, self.max)
            return x

        elif self.mode == 'log10':
            log_x = self.mean + self.std * z
            if self.min is not None or self.max is not None:
                log_x = log_x.clamp(self.min, self.max)
            return torch.pow(10, log_x)

        elif self.mode == 'logit':
            logit = self.mean + self.std * z
            s = torch.sigmoid(logit)
            return s * (self.max - self.min) + self.min

        raise ValueError(f'Invalid parameter mode: {self.mode}')

