import sys, os
import project


def main(argv):
    config = project.api.get_config(argv)
    run_name = config.pop('run_name', 'default')
    examples = project.api.get_examples(config['dataset'])
    project.api.run_training(examples, config['training'], run_name)


if __name__ == '__main__':
    main(sys.argv[1:])

