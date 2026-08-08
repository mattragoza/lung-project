import sys, os
import project


def main(argv):
    config = project.api.get_config(argv)
    examples = project.api.get_examples(config['dataset'])
    project.api.run_test_eval(examples, config['test_eval'])


if __name__ == '__main__':
    main(sys.argv[1:])

