# common/timer.py

import time


class Timer:

    def __init__(self):
        import torch
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.t_prev = time.perf_counter()

    def tick(self, sync: bool = True, unit_b: int = 2**30):
        import torch

        if sync:
            torch.cuda.synchronize()

        curr_alloc = torch.cuda.memory_allocated() / unit_b
        curr_resvd  = torch.cuda.memory_reserved() / unit_b
        peak_alloc = torch.cuda.max_memory_allocated() / unit_b
        peak_resvd  = torch.cuda.max_memory_reserved() / unit_b

        torch.cuda.reset_peak_memory_stats()

        t_curr = time.perf_counter()
        t_delta = t_curr - self.t_prev

        self.t_prev = t_curr

        return {
            't_delta': round(t_delta, 4),
            'curr_alloc': round(curr_alloc, 4),
            'curr_resvd': round(curr_resvd, 4),
            'peak_alloc': round(peak_alloc, 4),
            'peak_resvd': round(peak_resvd, 4)
        }

