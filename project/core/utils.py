import sys, inspect, argparse, time, random
import numpy as np
import torch

MAX_SEED = 2**32 - 1


def get_random_seed():
    time.sleep((time.time() % 1) / 1000)
    return int(time.time() * 1e6) % MAX_SEED


def set_random_seed(random_seed=None):
    if random_seed is None:
        random_seed = get_random_seed()
    print(f'Setting random seed to {random_seed}')
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)



def timer(sync):
    if sync:
        torch.cuda.synchronize()
    return time.time()


def as_bool(s):
    if isinstance(s, str):
        s = s.lower()
        if s in {'true', 't', '1'}:
            return True
        elif s in {'false', 'f', '0'}:
            return False
        else:
            raise ValueError(f'{repr(s)} is not a valid bool string')
    else:
        return bool(s)


def generate_argument_parser(func):

    # get full argument specification
    argspec = inspect.getfullargspec(func)
    args = argspec.args or []
    defaults = argspec.defaults or ()
    undefined = object() # sentinel object
    n_undefined = len(args) - len(defaults)
    defaults = (undefined,) * n_undefined + defaults

    # auto-generate argument parser
    parser = argparse.ArgumentParser()
    for name, default in zip(argspec.args, defaults):
        type_ = argspec.annotations.get(name, None)

        if default is undefined: # positional argument
            parser.add_argument(name, type=type_)

        elif default is False or default is True and type_ in {bool, None}: # flag
            parser.add_argument(
                f'--{name}', default=False, type=as_bool, help=f'[{default}]'
            )
        else: # optional argument
            if type_ is None and default is not None:
                type_ = type(default)
            parser.add_argument(
                f'--{name}', default=default, type=type_, help=f'[{default}]'
            )

    return parser


def main(func):
    '''
    Decorator for auto parsing arguments and calling the main function
    '''
    parent = inspect.stack()[1].frame
    __name__ = parent.f_locals.get('__name__')
    if __name__ == '__main__':

        # parse and display command line arguments
        parser = generate_argument_parser(func)
        kwargs = vars(parser.parse_args(sys.argv[1:]))
        print(kwargs)

        # call the main function
        func(**kwargs)

    return func
