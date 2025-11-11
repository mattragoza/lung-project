from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from .core import utils, transforms
from .core import metrics as mm


def _to_numpy(t):
    return t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)


def _eval(pred, target, weight=None, name=None, profile=None):
    if profile is None:
        profile = 'res' if name[0] == 'r' else name[0]
    metrics = mm.evaluate_metrics(pred, target, weight, profile)
    return utils.namespace(metrics, name) if name else metrics


class Evaluator:
    '''
    Attributes:
        example_rows[phase]: Stores one row per example of global metrics
        material_rows[phase]: Stores one row per (example, material) with metrics
    '''
    def __init__(self):
        self.example_rows = defaultdict(list)
        self.material_rows = defaultdict(list)

    @torch.no_grad()
    def evaluate(self, outputs, epoch, phase, batch):
        batch_size = len(outputs['example'])
        base = {
            'epoch': int(epoch),
            'phase': str(phase),
            'batch': int(batch),
            'loss':  float(outputs['loss'].item()),
        }
        for k in range(batch_size):
            ex = outputs['example'][k]
            mat_mask = _to_numpy(outputs['mask'][k]).reshape(-1, 1)
            E_true_vox = _to_numpy(outputs['E_true'][k]).reshape(-1, 1) # Pa
            E_pred_vox = _to_numpy(outputs['E_pred'][k]).reshape(-1, 1) # Pa

            sel = (mat_mask > 0)
            ex_row = {**base, 'subject': ex.subject}
            ex_row['num_voxels'] = int(np.count_nonzero(sel))
            ex_row |= _eval(E_pred_vox[sel], E_true_vox[sel], name='E_voxel')

            if 'pde' in outputs:
                pde_output = outputs['pde'][k]

                cells = _to_numpy(pde_output['cells'])
                vol_cells = _to_numpy(pde_output['vol_cells'])
                mat_cells = _to_numpy(pde_output['mat_cells'])
                E_true_cells = _to_numpy(pde_output['E_true_cells']) # Pa
                E_pred_vals = _to_numpy(pde_output['E_pred_values']) # Pa
                u_true_vals = _to_numpy(pde_output['u_true_values']) # meters
                u_pred_vals = _to_numpy(pde_output['u_pred_values']) # meters
                res_vals = _to_numpy(pde_output['res_values'])
        
                # true scalar fields are defined on mesh cells
                #   pred scalar fields may need to be mapped from nodes to cells
                if pde_output['scalar_degree'] == 0:
                    E_pred_cells = E_pred_vals
                elif pde_output['scalar_degree'] == 1:
                    E_pred_cells = transforms.node_to_cell_values(cells, E_pred_vals)

                # the vector fields are nodal, evaluate on cells for volume weighting
                assert pde_output['vector_degree'] == 1
                u_true_cells = transforms.node_to_cell_values(cells, u_true_vals)
                u_pred_cells = transforms.node_to_cell_values(cells, u_pred_vals)
                res_cells    = transforms.node_to_cell_values(cells, res_vals)

                ex_row |= _eval(E_pred_cells, E_true_cells, vol_cells, name='E_cell')
                ex_row |= _eval(u_pred_cells, u_true_cells, vol_cells, name='u_cell')
                ex_row |= _eval(res_cells, None, vol_cells, name='res_cell')

            self.example_rows[phase].append(ex_row)

            # next evaluate metrics grouped by material label

            for label in np.unique(mat_mask[sel]):
                sel = (mat_mask == label)
                mat_row = {**base, 'subject': ex.subject, 'material': int(label)}
                mat_row['num_voxels'] = int(np.count_nonzero(sel))
                mat_row |= _eval(E_pred_vox[sel], E_true_vox[sel], name='E_voxel')

                if 'pde' in outputs:
                    sel = (mat_cells == label)
                    vol_mat = vol_cells[sel]

                    mat_row['num_cells'] = int(np.count_nonzero(sel))
                    mat_row |= _eval(E_pred_cells[sel], E_true_cells[sel], vol_mat, name='E_cell')
                    mat_row |= _eval(u_pred_cells[sel], u_true_cells[sel], vol_mat, name='u_cell')
                    mat_row |= _eval(res_cells[sel], None, vol_mat, name='res_cell')

                self.material_rows[phase].append(mat_row)

    def phase_end(self, epoch, phase, out_dir=None, clear=True):
        ex_df  = pd.DataFrame(self.example_rows[phase])
        mat_df = pd.DataFrame(self.material_rows[phase])

        if out_dir:
            out_dir = Path(out_dir)
            out_dir.mkdir(exist_ok=True, parents=True)
            ex_path  = out_dir / 'example_metrics.csv'
            mat_path = out_dir / 'material_metrics.csv'

            for df, path in [(ex_df, ex_path), (mat_df, mat_path)]:
                if df.empty:
                    continue
                write_header = (not path.is_file() or path.stat.st_size == 0)
                df.to_csv(path, index=False, mode='a', header=write_header)

        if clear:
            self.example_rows[phase].clear()
            self.material_rows[phase].clear()

        return ex_df, mat_df


if False: # DEPRECATED
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

    def summarize(self):
        pass

    def save_metrics(self, path):
        self.metrics.to_csv(path)


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

