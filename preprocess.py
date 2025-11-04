import sys, os
import project

CONFIG = {
    'copdgene': {
        'dataset_cls': project.datasets.copdgene.COPDGeneDataset,
        'preprocess_fn': project.preprocessing.api.preprocess_copdgene,
        'default_root': 'data/COPDGene',
        'default_subj': '16514P'
    },
    'shapenet': {
        'dataset_cls': project.datasets.shapenet.ShapeNetDataset,
        'preprocess_fn': project.preprocessing.api.preprocess_shapenet,
        'default_root': 'data/ShapeNetSem',
        'default_subj': 'wss.101354f9d8dede686f7b08d9de913afe'
    }
}

def parse_args(argv):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--subject', default=None, help='Subject IDs to preprocess (comma-separated)')
    p.add_argument('--data_root', default=None, help='Dataset root directory')
    p.add_argument('--variant', default='TEST', help='Output directory name')
    p.add_argument('--config', help='Path to JSON configuration file')
    p.add_argument('--dry_run', action='store_true')
    return p.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    dataset_cls = CONFIG[args.dataset]['dataset_cls']
    run_preprocess = CONFIG[args.dataset]['preprocess_fn']

    data_root = args.data_root or CONFIG[args.dataset]['default_root']

    if args.subject:
        subjects = args.subject.split(',')
    else:
        subjects = [CONFIG[args.dataset]['default_subj']]

    print(subjects)

    if not os.path.isdir(data_root):
        raise RuntimeError(f'{data_root} is not a valid directory')

    ds = dataset_cls(data_root)

    config = project.core.fileio.load_json(args.config) if args.config else {}
    examples_cfg = config.get('examples', {})
    preprocess_cfg = config.get('preprocess', {})

    for ex in ds.examples(subjects, variant=args.variant, **examples_cfg):
        print(f'Preprocessing subject: {ex.subject}')
        project.core.utils.pprint(ex, max_depth=2, max_items=20)
        if not args.dry_run:
            run_preprocess(ex, preprocess_cfg)


if __name__ == '__main__':
    main(sys.argv[1:])

