from typing import List, Dict, Tuple, Any

from . import utils


def parse_args(argv: List[str]):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('config', help='path to config file (JSON/YAML)')
    p.add_argument(
        '--set',
        default=[],
        action='append',
        metavar='KEY=VAL',
        help='override specific config value(s)'
    )
    return p.parse_args(argv)


def apply_overrides(config, overrides) -> Dict[str, Any]:
    cfg = dict(config)
    for item in overrides:
        keys, val = parse_override(item)
        set_config_value(cfg, keys, val)
    return cfg


def parse_override(item: str) -> Tuple[List[str], Any]:
    import json, yaml
    k, v = item.split('=', 1)
    keys = k.strip().split('.')
    try:
        val = json.loads(v)
    except json.JSONDecodeError:
        val = yaml.safe_load(v)
    return keys, val


def set_config_value(config, keys, val):
    curr = config
    for k in keys[:-1]:
        if k not in curr:
            curr[k] = {}
        curr = curr[k]
    try:
        curr[keys[-1]] = val
    except Exception as exc:
        print(keys, val)
        raise exc


# legacy

def as_bool(val):
    if isinstance(val, str):
        val = val.lower()
        if val in {'true', 't', '1'}:
            return True
        elif val in {'false', 'f', '0'}:
            return False
        raise ValueError(f'Invalid boolean string: {val:r}')
    return bool(val)


def generate_argument_parser(func):
    import inspect, argparse

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
    import inspect
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

