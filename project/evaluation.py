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

        self.output_dir = Path('./outputs')
        self.output_dir.mkdir(exist_ok=True, parents=True)

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
            ex_row = {**base, 'subject': ex.subject}
            mat_labels = set()

            if 'mask' in outputs:
                mat_mask = _to_numpy(outputs['mask'][k]).reshape(-1, 1)
                E_true_vox = _to_numpy(outputs['E_true'][k]).reshape(-1, 1) # Pa
                E_pred_vox = _to_numpy(outputs['E_pred'][k]).reshape(-1, 1) # Pa

                sel = (mat_mask > 0)
                ex_row['num_voxels'] = int(np.count_nonzero(sel))
                ex_row |= _eval(E_pred_vox[sel], E_true_vox[sel], name='E_vox')

                mat_labels |= set(np.unique(mat_mask[sel]))

            if 'pde' in outputs:
                pde_output = outputs['pde'][k]
                vol_cells = _to_numpy(pde_output['volume'])
                mat_cells = _to_numpy(pde_output['material'])

                E_true_cells = _to_numpy(pde_output['E_true']) # Pa
                E_pred_cells = _to_numpy(pde_output['E_pred']) # Pa
                u_true_cells = _to_numpy(pde_output['u_true']) # meters
                u_pred_cells = _to_numpy(pde_output['u_pred']) # meters
                res_cells = _to_numpy(pde_output['residual'])
    
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


