import sys, os, argparse
import project


def run_optimize(ex, adapter_kws, solver_kws):
    from project.core import utils, fileio
    mesh = fileio.load_meshio(ex.paths['sim_fields'])
    utils.log(mesh)

    pde_adapter = project.physics.PhysicsAdapter(**adapter_kws)
    evaluator = projet.evaluation.Evaluator()


def parse_args(argv):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, help='Config file (JSON/YAML)')
    p.add_argument('--output', default=None, help='Output CSV path')
    return p.parse_args(argv)


def load_config(path):
    base = project.core.fileio.load_json(path)
    return {
        'examples': base.get('examples', {}),
        'optimize': base.get('optimize', {})
    }


def main(argv):
    args = parse_args(argv)
    print(vars(args))

    dataset_cls = CONFIG[args.dataset]['dataset_cls']
    data_root = args.data_root or CONFIG[args.dataset]['default_root']
    config = project.core.fileio.load_json(args.config) if args.config else {}
    print(config)

    out = dataset_cls(data_root='.')
    ds = dataset_cls(data_root)
    ds.load_metadata()
    
    rows = []
    for ex in ds.examples(subjects, args.variant, **config['examples']):
        print(f'Optimizing E field for subject: {ex.subject}')
        project.core.utils.pprint(ex, max_depth=2, max_items=20)
        output_path = out.path(ex.subject, variant='output', asset_type='mesh', mesh_tag='optimize')
        try:
            metrics = run_optimize(ex, output_path, config['optimize'])
            rows.append({
                'dataset': args.dataset,
                'subject': ex.subject,
                'variant': args.variant,
                'method': 'optimize',
                **metrics
            })
        except Exception as exc:
            utils.warn('ERROR: {exc}; skipping subject {ex.subject}')
            continue


if __name__ == '__main__':
    main(sys.argv[1:])

