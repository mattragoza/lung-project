import numpy as np
import pandas as pd


def _missing(s):
    return pd.isna(s) or str(s).strip() == ''


def _namespace(dct, name):
    return {f'{name}.{k}': v for k, v in dct.items()}


def _as_tuple(obj):
    return obj if isinstance(obj, tuple) else (obj,)


def _pass(metrics):
    reasons = metrics.get('reasons', ())
    return {**metrics, 'valid': len(reasons) == 0, 'reasons': reasons}


def _fail(metrics, reason):
    reasons = metrics.get('reasons', ()) + _as_tuple(reason)
    return {**metrics, 'valid': False, 'reasons': reasons}


def validate_example(
    ex, metadata: bool=True, paths: bool=True, artifacts: bool=True
):
    '''
    Args:
        ex: Example object from shapenet dataset
        metadata: If True, validate example metadata
        paths: If True, validate paths and file loading
        artifacts: If True, validate mesh and mask data
    Returns:
        Dict[str, Any] validation metrics
    '''
    metrics = {'subject': ex.subject}

    if metadata:
        m = validate_metadata(ex.metadata)
        metrics.update(_namespace(m, name='metadata'))

    if paths or artifacts:
        m, arts = validate_paths(ex.paths, load=True)
        metrics.update(_namespace(m, name='paths'))

    if artifacts:
        m = validate_artifacts(arts)
        metrics.update(_namespace(m, name='artifacts'))

    if metadata and not metrics['metadata.valid']:
        metrics = _fail(metrics, reason=metrics['metadata.reasons'])

    if paths and not metrics['paths.valid']:
        metrics = _fail(metrics, reason=metrics['paths.reasons'])

    if artifacts and not metrics['artifacts.valid']:
        metrics = _fail(metrics, reason=metrics['artifacts.reasons'])

    return _pass(metrics)


def validate_metadata(md):
    metrics = {}

    raw = md.get('raw')
    if _missing(raw):
        return _fail(metrics, reason='missing raw metadata')

    m = validate_category(raw.get('category'))
    metrics.update(_namespace(m, name='category'))

    m = validate_unit(raw.get('unit'))
    metrics.update(_namespace(m, name='unit'))

    m = validate_dims(raw.get('dims'))
    metrics.update(_namespace(m, name='dims'))

    if not metrics['category.valid']:
        metrics = _fail(metrics, reason=metrics['category.reasons'])

    if not metrics['unit.valid']:
        metrics = _fail(metrics, reason=metrics['unit.reasons'])

    if not metrics['dims.valid']:
        metrics = _fail(metrics, reason=metrics['dims.reasons'])

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
    metrics['len'] = len(metrics['set'])

    filtered = [c for c in cat if not c.startswith('_')]
    metrics['set_f'] = set(filtered)
    metrics['len_f'] = len(metrics['set_f'])

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

    # aligned.dims seems to be stored as (x,z,y) in cm
    metrics['xyz_m'] = dims[[0,2,1]] / 100.

    if not np.isfinite(dims).all():
        return _fail(metrics, reason='non-finite dim(s)')

    if (dims <= 0).any():
        return _fail(metrics, reason='non-positive dim(s)')

    metrics['prod']   = float(np.prod(dims))
    metrics['prod_m'] = float(np.prod(dims / 100.))

    return _pass(metrics)


def validate_paths(paths, load=False):
    from .core import fileio
    metrics, artifacts = {}, {}

    loader = fileio.load_trimesh if load else None
    m, a = validate_path(paths.get('source_mesh'), loader)
    metrics.update(_namespace(m, name='source_mesh'))
    artifacts['source_mesh'] = a

    loader = fileio.load_binvox if load else None
    m, a = validate_path(paths.get('source_mask'), loader)
    metrics.update(_namespace(m, name='source_mask'))
    artifacts['source_mask'] = a

    if not metrics['source_mesh.valid']:
        metrics = _fail(metrics, reason=metrics['source_mesh.reasons'])

    if not metrics['source_mask.valid']:
        metrics = _fail(metrics, reason=metrics['source_mask.reasons'])

    return _pass(metrics), artifacts


def validate_path(path, loader=None):
    metrics = {}
    loaded = None

    if _missing(path):
        return _fail(metrics, reason='missing path'), loaded

    metrics['exists'] = path.is_file()
    if not metrics['exists']:
        return _fail(metrics, reason='file does not exist'), loaded

    metrics['fsize'] = path.stat().st_size
    if metrics['fsize'] == 0:
        return _fail(metrics, reason='file size is zero'), loaded

    if loader:
        assert hasattr(loader, '__call__') # don't catch this
        try:
            loaded = loader(path)
        except Exception as e:
            metrics['exc'] = e
            return _fail(metrics, reason='failed to load'), loaded

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
    import trimesh
    metrics = {}

    if _missing(scene):
        return _fail(metrics, reason='missing scene')

    if not isinstance(scene, trimesh.Scene):
        return _fail(metrics, reason='not a trimesh scene')

    metrics['geometries'] = len(scene.geometry)
    if metrics['geometries'] == 1: # not fatal
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

    metrics['surface_area']  = float(mesh.area)

    # count boundary, interior, and non-manifold edges
    edge_types = surface_meshing.count_edge_types(mesh.faces)
    for k, v in edge_types.items():
        metrics[k + '_edges'] = v

    metrics['euler_number'] = mesh.euler_number 
    metrics['watertight']   = mesh.is_watertight
    metrics['components']   = len(mesh.split(only_watertight=False))

    if metrics['vertices'] < 4:
        return _fail(metrics, 'fewer than 4 vertices')

    metrics['volume'] = float(mesh.volume)

    if np.isclose(metrics['volume'], 0.):
        return _fail(metrics, 'zero volume')

    metrics['ch_volume'] = float(mesh.convex_hull.volume)
    metrics['convexity'] = metrics['volume'] / metrics['ch_volume']

    return _pass(metrics)


def validate_binvox_object(bv):
    from .preprocessing import mask_cleanup
    import binvox
    metrics = {}

    if _missing(bv):
        return _fail(metrics, reason='missing binvox')

    if not isinstance(bv, binvox.Binvox):
        return _fail(metrics, reason='not a binvox object')

    metrics['dims'] = np.asarray(bv.dims)
    metrics['translate'] = np.asarray(bv.translate)
    metrics['scale'] = float(bv.scale)
    metrics['axis_order'] = bv.axis_order

    try:
        array = bv.numpy()
    except Exception as e:
        metrics['exc'] = e
        return _fail(metrics, reason='failed to convert array')

    m = validate_binary_mask(array)
    metrics.update(_namespace(m, name='array'))

    try:
        cleaned = mask_cleanup.cleanup_binary_mask(array)
    except Exception as e:
        metrics['exc'] = e
        cleaned = None

    m = validate_binary_mask(cleaned)
    metrics.update(_namespace(m, name='clean'))

    if not (metrics['array.valid'] or metrics['clean.valid']):
        return _fail(metrics, reason=metrics['array.reasons'] + metrics['clean.reasons'])

    return _pass(metrics)


def validate_binary_mask(mask):
    from .preprocessing import mask_cleanup
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

    if metrics['nonzero'] == 1:
        return _fail(metrics, reason='one nonzero voxel')

    metrics['components'] = mask_cleanup.count_connected_components(mask)

    p5, p50, p95 = mask_cleanup.compute_thickness_metrics(mask, p=[5, 50, 95])
    metrics['thickness_p5']  = p5
    metrics['thickness_p50'] = p50
    metrics['thickness_p95'] = p95

    p5, p50, p95 = mask_cleanup.compute_cross_section_metrics(mask)
    metrics['area_p5']  = p5
    metrics['area_p50'] = p50
    metrics['area_p95'] = p95

    return _pass(metrics)

