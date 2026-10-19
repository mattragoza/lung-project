# common/timer.py

import time


class Timer:

    def tick(self, sync: bool = True, unit_b: int = 1024**3):
        import torch

        if sync:
            torch.cuda.synchronize()

        curr_alloc = torch.cuda.memory_allocated()
        curr_resvd = torch.cuda.memory_reserved()
        peak_alloc = torch.cuda.max_memory_allocated()
        peak_resvd = torch.cuda.max_memory_reserved()

        mem_free, mem_total = torch.cuda.mem_get_info()
        mem_used = mem_total - mem_free

        t_curr = time.perf_counter()
        t_prev = getattr(self, 't_prev', t_curr)
        self.t_prev = t_curr

        torch.cuda.reset_peak_memory_stats()

        return {
            't_delta': t_curr - t_prev,
            'curr_alloc': curr_alloc / unit_b,
            'curr_resvd': curr_resvd / unit_b,
            'peak_alloc': peak_alloc / unit_b,
            'peak_resvd': peak_resvd / unit_b,
            'mem_used': mem_used / unit_b,
            'mem_free': mem_free / unit_b,
            'mem_total': mem_total / unit_b
        }

