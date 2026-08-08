import sys, os
import project


def main(argv):
    config = project.api.get_config(argv)
    examples = project.api.get_examples(config['dataset'])
    project.api.run_preprocess(examples, config['preprocessing'])


if __name__ == '__main__':
    main(sys.argv[1:])

