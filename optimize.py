import sys, os
import pandas as pd
import project
import project.optimization
import warp as wp
wp.config.quiet = True


CONFIG = {
    'copdgene': {
        'dataset_cls': project.datasets.copdgene.COPDGeneDataset,
        'default_root': 'data/COPDGene',
        'default_subj': '16514P'
    },
    'shapenet': {
        'dataset_cls': project.datasets.shapenet.ShapeNetDataset,
        'default_root': 'data/ShapeNetSem',
        'default_subj': 'wss.101354f9d8dede686f7b08d9de913afe'
    }
}

def parse_args(argv):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--subject', default=None, help='Subject IDs to optimize (comma-separated)')
    p.add_argument('--data_root', default=None, help='Dataset root directory')
    p.add_argument('--variant', default='TEST', help='Preprocessed data variant')
    p.add_argument('--config', default=None, help='Path to JSON configuration file')
    p.add_argument('--output', default=None, help='Output csv path')
    return p.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    print(vars(args))

    dataset_cls = CONFIG[args.dataset]['dataset_cls']
    run_optimize = project.optimization.optimize_elasticity_field

    data_root = args.data_root or CONFIG[args.dataset]['default_root']

    if args.subject:
        subjects = args.subject.split(',')
    else:
        subjects = [CONFIG[args.dataset]['default_subj']]

    if not os.path.isdir(data_root):
        raise RuntimeError(f'{data_root} is not a valid directory')

    ds = dataset_cls(data_root)
    ds.load_metadata()

    out = dataset_cls(data_root='.')

    config = project.core.fileio.load_json(args.config) if args.config else {}
    examples_cfg = config.get('examples', {})
    optimize_cfg = config.get('optimize', {})
    print({'examples': examples_cfg, 'optimize': optimize_cfg})
    
    rows = []
    for ex in ds.examples(subjects, args.variant, **examples_cfg):
        print(f'Optimizing E field for subject: {ex.subject}')
        project.core.utils.pprint(ex, max_depth=2, max_items=20)
        output_path = out.path(ex.subject, variant='output', asset_type='mesh', mesh_tag='optimize')
        metrics = run_optimize(
            mesh_path=ex.paths['sim_fields'],
            output_path=output_path,
            unit_m=ex.metadata['unit'],
            **optimize_cfg
        )
        rows.append({
            'dataset': args.dataset,
            'subject': ex.subject,
            'variant': args.variant,
            'method': 'optimize',
            **metrics
        })

    df = pd.DataFrame(rows)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_rows', None)

    if len(subjects) == 1:
        print(df.T)

    if args.output:
        df.to_csv(args.output, index=False)


if __name__ == '__main__':
    main(sys.argv[1:])

