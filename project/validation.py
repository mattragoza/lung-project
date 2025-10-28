import numpy as np
import pandas as pd


def _missing(s):
    return pd.isna(s) or str(s).strip() == ''


def _namespace(dct, name):
    return {f'{name}.{k}': v for k, v in dct.items()}


def _as_list(obj):
    return obj if isinstance(obj, list) else [obj]


def _pass(metrics):
    reasons = metrics.get('reasons', [])
    return {**metrics, 'valid': len(reasons) == 0, 'reasons': reasons}


def _fail(metrics, reason):
    reasons = metrics.get('reasons', []) + _as_list(reason)
    return {**metrics, 'valid': False, 'reasons': reasons}


def validate_example(ex):
    metrics = {'subject': ex.subject}

    m = validate_metadata(ex.metadata)
    metrics.update(_namespace(m, name='metadata'))

    m, artifacts = validate_paths(ex.paths)
    metrics.update(_namespace(m, name='paths'))

    m = validate_artifacts(artifacts)
    metrics.update(_namespace(m, name='artifacts'))

    if not metrics['metadata.valid']:
        metrics = _fail(metrics, reason=metrics['metadata.reasons'])

    if not metrics['paths.valid']:
        metrics = _fail(metrics, reason=metrics['paths.reasons'])

    if not metrics['artifacts.valid']:
        metrics = _fail(metrics, reason=metrics['artifacts.reasons'])

    return _pass(metrics)


def validate_metadata(md):
    metrics = {}

    m = validate_category(md.get('category'))
    metrics.update(_namespace(m, name='category'))

    m = validate_unit(md.get('unit'))
    metrics.update(_namespace(m, name='unit'))

    m = validate_dims(md.get('aligned.dims'))
    metrics.update(_namespace(m, name='dims'))

    if not metrics['category.valid']:
        metrics = _fail(metrics, reason=metrics['category.reasons'])

    if not metrics['unit.valid']:
        metrics = _fail(metrics, reason=metrics['unit.reasons'])

    if not metrics['dims.valid']:
        metrics = _fail(metrics, reason=metrics['dims.reasons'])

    if metrics['dims.valid'] and metrics['unit.valid']:
        metrics['dims_by_unit'] = metrics['dims.value'] / metrics['unit.value']

    return _pass(metrics)


def validate_category(cat_raw):
    from .datasets.shapenet import _parse_category
    metrics = {}

    if _missing(cat_raw):
        return _fail(metrics, reason='missing category')

    try:
        cat = _parse_category(cat_raw)
    except Exception as e:
        metrics['exc'] = e
        return _fail(metrics, reason='failed to parse category')

    metrics['set'] = set(cat)
    metrics['len'] = len(cat)

    return _pass(metrics)


def validate_unit(unit_raw):
    metrics = {}

    if _missing(unit_raw):
        return _fail(metrics, reason='missing unit')

    try:
        unit = float(unit_raw)
    except Exception as e:
        metrics['exc'] = e
        return _fail(metrics, reason='failed to parse unit')

    metrics['value'] = unit

    if not np.isfinite(unit):
        return _fail(metrics, reason='non-finite unit')

    if unit <= 0:
        return _fail(metrics, reason='non-positive unit')

    return _pass(metrics)


def validate_dims(dims_raw):
    from .datasets.shapenet import _parse_aligned_dims
    metrics =  {}

    if _missing(dims_raw):
        return _fail(metrics, reason='missing dims')

    try:
        dims = _parse_aligned_dims(dims_raw)
    except Exception as e:
        metrics['exc'] = e
        return _fail(metrics, reason='failed to parse dims')

    metrics['shape'] = dims.shape
    metrics['value'] = dims

    if dims.shape != (3,):
        return _fail(metrics, reason='invalid dims shape')

    if not np.isfinite(dims).all():
        return _fail(metrics, reason='non-finite dim(s)')

    if (dims <= 0).any():
        return _fail(metrics, reason='non-positive dim(s)')

    metrics['product'] = float(np.prod(dims))

    return _pass(metrics)


def validate_paths(paths):
    from .core import fileio
    metrics, artifacts = {}, {}

    m, a = validate_path(paths.get('source_mesh'), fileio.load_trimesh)
    metrics.update(_namespace(m, name='source_mesh'))
    artifacts['source_mesh'] = a

    m, a = validate_path(paths.get('source_mask'), fileio.load_binvox)
    metrics.update(_namespace(m, name='source_mask'))
    artifacts['source_mask'] = a

    if not metrics['source_mesh.valid']:
        metrics = _fail(metrics, reason=metrics['source_mesh.reasons'])

    if not metrics['source_mask.valid']:
        metrics = _fail(metrics, reason=metrics['source_mask.reasons'])

    return _pass(metrics), artifacts


def validate_path(path, loader):
    metrics = {}

    if _missing(path):
        return _fail(metrics, reason='missing path'), None

    metrics['exists'] = path.is_file()
    if not metrics['exists']:
        return _fail(metrics, reason='file does not exist'), None

    metrics['fsize'] = path.stat().st_size
    if metrics['fsize'] == 0:
        return _fail(metrics, reason='file size is zero'), None

    try:
        loaded = loader(path)
    except Exception as e:
        metrics['exc'] = e
        return _fail(metrics, reason='failed to load'), None

    return _pass(metrics), loaded


def validate_artifacts(artifacts):
    metrics = {}

    if _missing(artifacts):
        return _fail(metrics, reason='missing artifacts')

    m = validate_trimesh_scene(artifacts.get('source_mesh'))
    metrics.update(_namespace(m, name='scene'))

    m = validate_binvox_object(artifacts.get('source_mask'))
    metrics.update(_namespace(m, name='binvox'))

    if not metrics['scene.valid']:
        metrics = _fail(metrics, reason=metrics['scene.reasons'])

    if not metrics['binvox.valid']:
        metrics = _fail(metrics, reason=metrics['binvox.reasons'])

    return _pass(metrics)


def validate_trimesh_scene(scene):
    from .preprocessing import surface_meshing
    metrics = {}

    if _missing(scene):
        return _fail(metrics, reason='missing scene')

    metrics['geometries'] = len(scene.geometry)
    if metrics['geometries'] == 1:
        metrics = _fail(metrics, reason='single geometry')

    try:
        mesh = scene.to_mesh()
    except Exception as e:
        metrics['exc'] = e
        return _fail(metrics, reason='failed to convert mesh')

    m = validate_triangular_mesh(mesh)
    metrics.update(_namespace(m, name='mesh'))

    try:
        repaired = surface_meshing.repair_surface_mesh(mesh)
    except Exception as e:
        metrics['exc'] = e
        repaired = None

    m = validate_triangular_mesh(repaired)
    metrics.update(_namespace(m, name='repair'))

    if not (metrics['mesh.valid'] or metrics['repair.valid']):
        metrics = _fail(metrics, reason=metrics['mesh.reasons'] + metrics['repair.reasons'])

    return _pass(metrics)


def validate_triangular_mesh(mesh):
    from .preprocessing import surface_meshing
    metrics = {}

    if _missing(mesh):
        return _fail(metrics, reason='missing mesh')

    metrics['vertices'] = len(mesh.vertices)
    if metrics['vertices'] == 0:
        return _fail(metrics, reason='zero vertices')

    bbox_min = mesh.vertices.min(axis=0)
    bbox_max = mesh.vertices.max(axis=0)
    extent = bbox_max - bbox_min

    metrics['bbox_min'] = np.array(bbox_min)
    metrics['bbox_max'] = np.array(bbox_max)
    metrics['extent']   = np.array(extent)

    metrics['faces'] = len(mesh.faces)
    if metrics['faces'] == 0:
        return _fail(metrics, reason='zero faces')

    # count boundary, interior, and non-manifold edges
    edge_types = surface_meshing.count_edge_types(mesh.faces)
    for k, v in edge_types.items():
        metrics[k + '_edges'] = v

    metrics['euler_number'] = mesh.euler_number 
    metrics['is_watertight'] = mesh.is_watertight
    #metrics['components'] = len(mesh.split(only_watertight=False))
    metrics['surface_area'] = float(mesh.area)
    metrics['volume'] = float(mesh.volume)

    if np.isclose(metrics['volume'], 0.):
        return _fail(metrics, 'zero volume')

    metrics['convexity'] = metrics['volume'] / float(mesh.convex_hull.volume)

    return _pass(metrics)


def validate_binvox_object(bv):
    import binvox
    metrics = {}

    if _missing(bv):
        return _fail(metrics, reason='missing binvox')

    if not isinstance(bv, binvox.Binvox):
        return _fail(metrics, reason='not a binvox object')

    metrics['translate'] = np.asarray(bv.translate)
    metrics['scale'] = float(bv.scale)

    try:
        array = bv.numpy()
    except Exception as e:
        metrics['exc'] = e
        return _fail(metrics, reason='failed to convert array')

    m = validate_binary_mask(array)
    metrics.update(_namespace(m, name='array'))

    if not metrics['array.valid']:
        return _fail(metrics, reason=metrics['array.reasons'])

    return _pass(metrics)


def validate_binary_mask(mask):
    metrics = {}

    if not isinstance(mask, np.ndarray):
        return _fail(metrics, reason='not a numpy array')

    metrics['ndim'] = mask.ndim
    metrics['shape'] = mask.shape
    metrics['dtype'] = mask.dtype

    if mask.ndim != 3 or min(mask.shape) <= 1:
        return _fail(metrics, reason='invalid shape')

    metrics['min'] = float(mask.min())
    metrics['max'] = float(mask.max())
    metrics['sum'] = float(mask.sum())
    metrics['mean'] = float(mask.mean())
    metrics['nonzero'] = int(np.count_nonzero(mask))
    metrics['nans'] = int(np.count_nonzero(np.isnan(mask)))

    if metrics['nans'] > 0:
        return _fail(metrics, reason='contains nans')

    if metrics['nonzero'] == 0:
        return _fail(metrics, reason='all zero voxels')

    return _pass(metrics)

