from typing import List, Dict, Tuple, Any

from . import utils, fileio


def get_config_from_argv(argv: List[str]):
    args = parse_args(argv)
    config = fileio.load_config(args.config)
    config = apply_overrides(config, args.set)
    utils.pprint(config)
    return config


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


def apply_overrides(config, overrides):
    cfg = dict(config)
    for item in overrides:
        keys, val = parse_override(item)
        set_config_value(cfg, keys, val)
    return cfg


def parse_override(item: str):
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
    except Exception as e:
        print(keys, val)
        raise

