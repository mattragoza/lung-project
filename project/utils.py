import sys, inspect, argparse, time
import numpy as np
import xarray as xr
import torch


def as_xarray(a, dims=None, coords=None, name=None):
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    if dims is None:
        dims = [f'dim{i}' for i in range(a.ndim)]
    if coords is None:
        coords = {d: np.arange(a.shape[i]) for i, d in enumerate(dims)}
    return xr.DataArray(a, dims=dims, coords=coords, name=name)


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


def generate_arg_parser(func):

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

        elif default is False and type_ in {bool, None}: # flag
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
    parent_frame = inspect.stack()[1].frame
    __name__ = parent_frame.f_locals.get('__name__')

    if __name__ == '__main__':

        # parse and display command line arguments
        parser = generate_arg_parser(func)
        kwargs = vars(parser.parse_args(sys.argv[1:]))
        print(kwargs)

        # call the main function
        func(**kwargs)

    return func
