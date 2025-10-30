import sys, os, json
import project.datasets
import project.preprocessing

CONFIG = {
    'copdgene': {
        'dataset_cls': project.datasets.copdgene.COPDGeneDataset,
        'pipeline_fn': project.preprocessing.api.preprocess_copdgene,
        'default_root': 'data/COPDGene',
        'default_subj': '16514P'
    },
    'shapenet': {
        'dataset_cls': project.datasets.shapenet.ShapeNetDataset,
        'pipeline_fn': project.preprocessing.api.preprocess_shapenet,
        'default_root': 'data/ShapeNetSem',
        'default_subj': 'wss.101354f9d8dede686f7b08d9de913afe'
    }
}

def parse_args(argv):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--data_root', default=None, help='Dataset root directory')
    p.add_argument('--subject', nargs='*', help='Subject IDs to preprocess')
    p.add_argument('--variant', default='TEST', help='Output subdirectory name')
    p.add_argument('--config', help='Path to JSON configuration file')
    p.add_argument('--dry_run', action='store_true')
    return p.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    dataset_cls = CONFIG[args.dataset]['dataset_cls']
    run_pipeline = CONFIG[args.dataset]['pipeline_fn']

    data_root = args.data_root or CONFIG[args.dataset]['default_root']
    subjects = args.subject or [CONFIG[args.dataset]['default_subj']]

    if not os.path.isdir(data_root):
        raise RuntimeError(f'{data_root} is not a valid directory')

    ds = dataset_cls(data_root)
    config = json.load(args.config) if args.config else {}

    for ex in ds.examples(subjects, args.variant, **config.get('examples', {})):
        project.core.utils.pprint(ex, max_depth=2, max_items=20)
        if not args.dry_run:
            run_pipeline(ex, config.get('pipeline', {}))


if __name__ == '__main__':
    main(sys.argv[1:])

