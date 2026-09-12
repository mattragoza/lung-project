# cli.py

from typing import List, Dict, Tuple, Any

import argparse


def resolve_config(argv: List[str]) -> Dict[str, Any]:
    from .common import fileio
    args = parse_args(argv)
    config = fileio.load_config(args.config)
    return apply_overrides(config, args.set)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='path to config file (JSON/YAML)')
    parser.add_argument(
        '--set',
        default=[],
        action='append',
        metavar='KEY=VAL',
        help='override specific config value(s)'
    )
    return parser.parse_args(argv)


def apply_overrides(config: dict, overrides: List[str]) -> Dict[str, Any]:
    config = dict(config)
    for string in overrides:
        keys, value = parse_override(string)
        set_config_value(config, keys, value)
    return config


def parse_override(string: str) -> Tuple[List[str], Any]:
    import json, yaml

    raw_keys, raw_value = string.split('=', 1)

    keys = raw_keys.strip().split('.')
    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        value = yaml.safe_load(raw_value)

    return keys, value


def set_config_value(config: dict, keys: List[str], value: Any):
    if len(keys) < 1:
        raise ValueError('No keys were provided')

    scope = config
    for key in keys[:-1]:
        if key not in scope:
            scope[key] = {}
        scope = scope[key]

    try:
        scope[keys[-1]] = value
    except Exception:
        print(keys, value)
        raise


# DEPRECATED


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

