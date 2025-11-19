from typing import List, Dict, Any
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
        profile = name.split('_')[0]
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

        self.output_dir = Path('./outputs')
        self.output_dir.mkdir(exist_ok=True, parents=True)

    @torch.no_grad()
    def evaluate(self, outputs: Dict[str, Any], epoch: int, phase: str, batch: int):
        batch_size = len(outputs['example'])
        base = {
            'epoch': int(epoch),
            'phase': str(phase),
            'batch': int(batch),
            'loss':  float(outputs['loss'].item()),
        }
        for k in range(batch_size):
            ex = outputs['example'][k]
            ex_row = {**base, 'subject': ex.subject}
            mat_labels = set()

            if 'mask' in outputs: # voxel domain
                mat_mask = _to_numpy(outputs['mask'][k]).reshape(-1, 1)
                E_true_vox = _to_numpy(outputs['E_true'][k]).reshape(-1, 1) # Pa
                E_pred_vox = _to_numpy(outputs['E_pred'][k]).reshape(-1, 1) # Pa

                sel = (mat_mask > 0)
                ex_row['num_voxels'] = int(np.count_nonzero(sel))
                ex_row |= _eval(E_pred_vox[sel], E_true_vox[sel], name='E_vox')

                mat_labels |= set(np.unique(mat_mask[sel]))

            if 'pde' in outputs: # mesh domain
                pde_output = outputs['pde'][k]
                vol_cells = _to_numpy(pde_output['volume'])
                mat_cells = _to_numpy(pde_output['material'].cells)

                rho_true_cells = _to_numpy(pde_output['rho_true'].cells)
                rho_pred_cells = _to_numpy(pde_output['rho_pred'].cells)
                E_true_cells = _to_numpy(pde_output['E_true'].cells) # Pa
                E_pred_cells = _to_numpy(pde_output['E_pred'].cells) # Pa
                u_true_cells = _to_numpy(pde_output['u_true'].cells) # meters
                u_pred_cells = _to_numpy(pde_output['u_pred'].cells) # meters
                res_cells = _to_numpy(pde_output['residual'].cells)
    
                ex_row |= _eval(rho_pred_cells, rho_true_cells, vol_cells, name='rho_cell')
                ex_row |= _eval(E_pred_cells, E_true_cells, vol_cells, name='E_cell')
                ex_row |= _eval(u_pred_cells, u_true_cells, vol_cells, name='u_cell')
                ex_row |= _eval(res_cells, None, vol_cells, name='res_cell')

                mat_labels |= set(np.unique(mat_cells))

            ex_row['num_materials'] = len(mat_labels)
            self.example_rows[phase].append(ex_row)

            # next, group by material label
            for label in sorted(mat_labels):
                mat_row = {**base, 'subject': ex.subject, 'material': int(label)}

                if 'mask' in outputs:
                    sel = (mat_mask == label)
                    mat_row['num_voxels'] = int(np.count_nonzero(sel))
                    mat_row |= _eval(E_pred_vox[sel], E_true_vox[sel], name='E_vox')

                if 'pde' in outputs:
                    sel = (mat_cells == label)
                    num_cells = int(np.count_nonzero(sel))
                    if num_cells == 0:
                        continue
                    vol_mat = vol_cells[sel]
                    if np.sum(vol_mat) <= 0:
                        utils.warn('WARNING: Zero-sum cell volume for subject {ex.subject}, material {label}; skipping.')
                        continue

                    mat_row['num_cells'] = num_cells
                    mat_row |= _eval(rho_pred_cells[sel], rho_true_cells[sel], vol_mat, name='rho_cell')
                    mat_row |= _eval(E_pred_cells[sel], E_true_cells[sel], vol_mat, name='E_cell')
                    mat_row |= _eval(u_pred_cells[sel], u_true_cells[sel], vol_mat, name='u_cell')
                    mat_row |= _eval(res_cells[sel], None, vol_mat, name='res_cell')

                self.material_rows[phase].append(mat_row)

    def phase_end(self, epoch, phase):

        self.output_dir.mkdir(parents=True, exist_ok=True)
        ex_path  = self.output_dir / 'example_metrics.csv'
        mat_path = self.output_dir / 'material_metrics.csv'

        ex_df_all = pd.concat(
            [pd.DataFrame(rows) for rows in self.example_rows.values()],
            ignore_index=True
        )
        mat_df_all = pd.concat(
            [pd.DataFrame(rows) for rows in self.material_rows.values()],
            ignore_index=True
        )
        for df, path in [(ex_df_all, ex_path), (mat_df_all, mat_path)]:
            if df.empty:
                continue
            tmp = path.with_suffix(path.suffix + '.tmp')
            try:
                utils.log(f'Saving {path}')
                df.to_csv(tmp, index=False)
                tmp.replace(path)
            finally:
                tmp.unlink(missing_ok=True)

        # current phase metrics
        return pd.DataFrame(self.example_rows[phase])

