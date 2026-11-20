import sys, os
import numpy as np
import pandas as pd

import project
import param_search as ps

TRAIN_METHOD = 'Ours'
OPTIM_METHOD = 'Baseline'


class Experiment:

    def __init__(self, base_prefix, protocol_keys, subject_split):
        from IPython.display import display

        self.base_prefix = base_prefix
        self.protocol_keys = protocol_keys
        self.subject_split = subject_split

        print('Loading job metadata')
        self.train_jobs = load_train_jobs(self.base_prefix + '_train', self.protocol_keys)
        self.optim_jobs = load_optim_jobs(self.base_prefix + '_optimize', self.protocol_keys)

        print('Loading evaluation metrics')
        self.train_metrics = load_train_metrics(self.train_jobs)
        self.optim_metrics = load_optim_metrics(self.optim_jobs)

        print('Selecting best train epochs')
        self.best_metrics, self.best_lookup = select_best_epoch(
            self.train_metrics, self.protocol_keys
        )
        display(self.best_lookup)

        # NOTE: this does not work for train phase (num_voxels is missing)
        self.subject_lookup = (
            self.optim_metrics[['num_voxels', 'params.subject']]
            .dropna()
            .drop_duplicates()
            .set_index('num_voxels', verify_integrity=True)['params.subject']
        )

        self.images, self.masks = {}, {}
        self.E_train, self.E_optim = {}, {}
        self.region_rows, self.rank_rows = {}, {}

    def combine_metrics(self, phase):

        combined = pd.concat([
            self.best_metrics[
                self.best_metrics['phase'].eq(phase)
            ].assign(method=TRAIN_METHOD),
            self.optim_metrics[
                self.optim_metrics['params.subject'].isin(self.subject_split[phase])
            ].assign(method=OPTIM_METHOD)
        ])

        combined['subject'] = combined['num_voxels'].map(self.subject_lookup)

        has_subject = combined['subject'].notna()
        if not has_subject.all():
            print(f'WARNING: {(~has_subject).sum()} rows have unresolved subject IDs', file=sys.stderr)

        return combined

    def load_assets(self, phase):
        print('Loading common assets')

        default_job = self.train_jobs.iloc[0]
        subject_set = self.subject_split[phase]
        config = get_eval_config(default_job, subject_set)

        for ex in project.get_examples(config['dataset']):
            self.images[ex.subject] = load_image(ex) 
            self.masks[ex.subject] = load_masks(ex)

    def evaluate_protocol(self, protocol_vals, phase):
        from tqdm.notebook import tqdm
        print(f'Evaluating protocol: {protocol_vals}')

        if isinstance(protocol_vals, list):
            protocol_vals = tuple(protocol_vals)
        elif not isinstance(protocol_vals, tuple):
            protocol_vals = (protocol_vals,)

        subject_set = self.subject_split[phase]
        best = self.best_lookup.loc[protocol_vals]
        
        train_job = self.train_jobs.loc[*protocol_vals, best['params.random_seed']]
        trainer = get_trainer(train_job, subject_set)
        trainer.load_checkpoint(epoch=int(best.epoch))

        E_train, E_optim = {}, {}
        region_rows, rank_rows = [], []
        
        for batch in tqdm(trainer.train_loader):
            subject = batch['example'][0].subject
            optim_job = self.optim_jobs.loc[*protocol_vals, subject]

            try:
                E_optim[subject] = load_baseline(optim_job)
            except FileNotFoundError as e:
                print(f'WARNING: Skipping {subject}: {e}')
                continue

            E_train[subject] = run_inference(trainer, batch)

            region_rows.extend(evaluate_regions(
                E_train[subject], self.masks[subject],
                subject=subject, method=TRAIN_METHOD
            ))
            region_rows.extend(evaluate_regions(
                E_optim[subject], self.masks[subject],
                subject=subject, method=OPTIM_METHOD
            ))
            rank_rows.append(evaluate_ranking(
                E_train[subject], self.masks[subject],
                subject=subject, method=TRAIN_METHOD
            ))
            rank_rows.append(evaluate_ranking(
                E_optim[subject], self.masks[subject],
                subject=subject, method=OPTIM_METHOD
            ))

        del trainer

        self.E_train[protocol_vals] = E_train
        self.E_optim[protocol_vals] = E_optim

        self.region_rows[protocol_vals] = region_rows
        self.rank_rows[protocol_vals] = rank_rows

    def evaluate_all(self, phase):
        for protocol_vals, _ in self.best_lookup.iterrows():
            self.evaluate_protocol(protocol_vals, phase)

    def get_region_eval(self):
        records = []
        for protocol_vals, region_rows in self.region_rows.items():
            protocol = dict(zip(self.protocol_keys, protocol_vals, strict=True))
            records.extend(protocol | row for row in region_rows)
        return pd.DataFrame.from_records(records)

    def get_rank_eval(self):
        records = []
        for protocol_vals, rank_rows in self.rank_rows.items():
            protocol = dict(zip(self.protocol_keys, protocol_vals, strict=True))
            records.extend(protocol | row for row in rank_rows)
        return pd.DataFrame.from_records(records)


def load_job_config(job):
    import pathlib, shlex
    script = pathlib.Path(job.script_path).read_text().replace('\\\n', ' ')
    command = next(l.strip() for l in script.splitlines() if l.startswith('python '))
    return project.api.get_config(shlex.split(command)[2:])


def load_optim_jobs(base_dir, protocol_keys, subject_key='params.subject'):
    index_cols = [*protocol_keys, subject_key]
    jobs = ps.load(base_dir)
    if 'params.E_beta' in jobs:
        jobs['params.E_beta'] = jobs['params.E_beta'].astype('string').fillna('hard clamp')
    return jobs.set_index(index_cols, verify_integrity=True)


def load_train_jobs(base_dir, protocol_keys, seed_key='params.random_seed'):
    index_cols = [*protocol_keys, seed_key]
    jobs = ps.load(base_dir)
    if 'params.E_beta' in jobs:
        jobs['params.E_beta'] = jobs['params.E_beta'].astype('string').fillna('hard clamp')
    return jobs.set_index(index_cols, verify_integrity=True)


def load_optim_metrics(jobs):
    return ps.outputs(
        jobs.reset_index(),
        meta_cols=['base_dir', 'job_name', 'job_id'],
        output_name='outputs/optimize/default/eval_metrics.csv',
        read_csv_kws={'index_col': 0},
        skip_errors=True
    )


def load_train_metrics(jobs):
    return ps.outputs(
        jobs.reset_index(),
        meta_cols=['base_dir', 'job_name', 'job_id'],
        output_name='outputs/training/default/eval_metrics.csv',
        skip_errors=True
    )


def select_best_epoch(
    train_metrics,
    protocol_keys,
    seed_key='params.random_seed',
    epoch_key='epoch',
    phase_key='phase',
    metric_key='loss'
):
    # select best training (seed, epoch) by val performance
    epoch_keys = list(protocol_keys) + [seed_key, epoch_key]

    epoch_scores = (
        train_metrics[train_metrics[phase_key].eq('val')]
        .groupby(epoch_keys, dropna=False)[metric_key].mean()
        .reset_index()
    )
    best_epochs = epoch_scores.loc[
        epoch_scores.groupby(protocol_keys, dropna=False)[metric_key].idxmin()
    ]
    best_lookup = best_epochs.set_index(
        protocol_keys, verify_integrity=True
    )
    
    # filter metrics to best (seed, epoch) per protocol
    best_metrics = train_metrics.merge(
        best_epochs[epoch_keys],
        on=epoch_keys,
        how='inner',
        validate='many_to_one'
    )

    return best_metrics, best_lookup


# ----- inference helpers -----


def get_eval_config(train_job, subject_list):
    from copy import deepcopy
    config = deepcopy(load_job_config(train_job))
    config['dataset']['subjects'] = subject_list
    config['training']['split']['method'] = None
    config['training']['loaders']['shuffle'] = False
    config['training']['loaders']['batch_size'] = 1
    return config


def get_trainer(train_job, subject_list):
    from pathlib import Path
    import project.training
    config = get_eval_config(train_job, subject_list)
    examples = project.get_examples(config['dataset'])
    output_dir = Path(train_job.work_dir) / 'outputs'
    trainer = project.training.get_trainer(
        examples, config['training'], output_dir=output_dir
    )
    trainer.model.eval()
    return trainer


def run_inference(trainer, batch):
    outputs = trainer.run_batch(batch, train_mode=False, eval_mode=True)
    return outputs['preds']['E'][0,0].detach().cpu().numpy()


def load_baseline(optim_job, which='optimize'):
    from pathlib import Path
    import numpy as np
    subject = optim_job.name[-1]
    output_dir = Path(optim_job.work_dir) / 'outputs'
    path = output_dir / which / 'default' / subject / 'rasters' / 'E_pred.nii.gz'
    return project.fileio.load_nibabel(path).get_fdata(dtype=np.float32)


def load_image(ex, state='curr_state'):
    path = ex.paths[state]['resampled_image']
    return project.fileio.load_nibabel(path).get_fdata()


def load_masks(ex):
    masks = {}
    for path in ex.paths['init_state']['segment_dir'].iterdir():
        if not str(path).endswith('nii.gz'):
            continue
        name = path.name.split('.')[0]
        if name.startswith('gt'): # skip high-attenuation masks
            continue
        masks[name] = project.fileio.load_nibabel(path).get_fdata() > 0

    masks['lobes'] = (
        masks.pop('lung_upper_lobe_left') |
        masks.pop('lung_lower_lobe_left') | 
        masks.pop('lung_upper_lobe_right') | 
        masks.pop('lung_middle_lobe_right') | 
        masks.pop('lung_lower_lobe_right')
    )
    masks['airways'] = masks.pop('lung_airways')
    masks['vessels'] = masks.pop('lung_arteries') | masks.pop('lung_veins')

    masks['parenchyma'] = masks['lobes'] & ~(masks['vessels'] | masks['airways'])

    for key in ['emphysema', 'ggo', 'consolidation', 'nodule', 'honeycombing', 'pleural_effusion', 'lt-950', 'lt-910', 'lt-850']:
        masks[key] = masks['parenchyma'] & masks[key]
    
    masks['abnormal'] = (
        masks['emphysema'] |
        masks['ggo'] | 
        masks['consolidation'] |
        masks['nodule'] |
        masks['honeycombing'] |
        masks['pleural_effusion']
    )
    masks['normal'] = masks['parenchyma'] & ~masks['abnormal']

    masks['harder'] = masks['parenchyma'] & (
        masks['ggo'] | masks['consolidation'] | masks['nodule']
    )
    masks['softer'] = masks['parenchyma'] & (
        masks['emphysema']
    )

    masks['ambiguous'] = masks['harder'] & masks['softer']
    masks['harder'] = masks['harder'] & ~masks['ambiguous']
    masks['softer'] = masks['softer'] & ~masks['ambiguous']

    return masks


def evaluate_regions(E_pred, masks, **metadata):
    rows = []
    for region, mask in masks.items():
        values = E_pred[mask > 0]
        finite = values[np.isfinite(values)]
        rows.append({
            **metadata,
            'region': region,
            'n_voxels': values.size,
            'n_finite': finite.size,
            'mean': finite.mean() if finite.size else np.nan,
            'std': finite.std() if finite.size else np.nan,
            'median': np.median(finite) if finite.size else np.nan,
            'Q1': np.quantile(finite, 0.25) if finite.size else np.nan,
            'Q3': np.quantile(finite, 0.75) if finite.size else np.nan,
            'min': finite.min() if finite.size else np.nan,
            'max': finite.max() if finite.size else np.nan
        })
    return rows
    

def evaluate_ranking(E_pred, masks, **metadata):
    E_normal = E_pred[masks['normal']]
    E_harder = E_pred[masks['harder']]
    E_softer = E_pred[masks['softer']]

    return {
        **metadata,
        'n_normal': np.isfinite(E_normal).sum(),
        'n_harder': np.isfinite(E_harder).sum(),
        'n_softer': np.isfinite(E_softer).sum(),
        'AUC_hard': probability_greater(E_harder, E_normal),
        'AUC_soft': probability_greater(E_normal, E_softer),
    }


def probability_greater(a, b):
    import scipy.stats

    a, b = a.ravel(), b.ravel()    
    if not a.size or not b.size:
        return np.nan
    
    # Mann-Whitney U statistic (Wilcoxon rank-sum test)
    ranks = scipy.stats.rankdata(np.concatenate([a, b]))
    u = ranks[:a.size].sum() - a.size * (a.size + 1) / 2

    # P(a > b) + 1/2 * P(a == b)
    return u / (a.size * b.size)

