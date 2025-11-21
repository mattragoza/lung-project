import sys, os
import project


def main(argv):
    config = project.core.cli.get_config_from_argv(argv)
    examples = project.api.get_examples(config['dataset'])
    project.api.run_preprocess(examples, config['preprocessing'])


if __name__ == '__main__':
    main(sys.argv[1:])

