import sys, os
import project


def main(argv):
    config = project.api.get_config(argv)
    examples = project.api.get_examples(config['dataset'])
    project.api.run_training(examples, config['training'])


if __name__ == '__main__':
    main(sys.argv[1:])

