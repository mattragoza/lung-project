import sys, os
sys.path.insert(0, os.environ['LP'])
import project


def main(argv):
    config = project.api.get_config(argv)
    examples = project.api.get_examples(config['dataset'])
    project.api.run_optimize(examples, config['optimization'])


if __name__ == '__main__':
    main(sys.argv[1:])

