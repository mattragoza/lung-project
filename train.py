import sys, os, argparse
import project


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True, help='Config file (JSON or YAML)')
    p.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    p.add_argument('--supervised', type=project.core.utils.as_bool, default=False)
    p.add_argument('--learning_rate', type=float, default=None)
    p.add_argument('--tv_reg_weight', type=float, default=None)
    p.add_argument('--output', type=str, default=None, help='Output CSV path')
    return p.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    config = project.core.fileio.load_config(args.config)
    examples = project.api.get_examples(config['dataset'])
    project.api.run_training(examples, config['training'])


if __name__ == '__main__':
    main(sys.argv[1:])

