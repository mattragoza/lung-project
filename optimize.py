import sys, os, argparse
import project


def parse_args(argv):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True, help='Path to config file')
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
    config = project.core.fileio.load_config(args.config)
    examples = project.api.get_examples(config['dataset'])
    project.api.run_optimize(examples, config['optimization'], args.output)


if __name__ == '__main__':
    main(sys.argv[1:])

