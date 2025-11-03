import numpy as np

from . import utils


def load_nibabel(path):
    import nibabel as nib
    utils.log(f'Loading {path}')
    return nib.load(path)


def save_nibabel(path, array, affine):
    import nibabel as nib
    utils.log(f'Saving {path}')
    nifti = nib.nifti1.Nifti1Image(array, affine)
    nib.save(nifti, path)


def load_simpleitk(path):
    import SimpleITK as sitk
    utils.log(f'Loading {path}')
    return sitk.ReadImage(path)


def save_simpleitk(path, image):
    import SimpleITK as sitk
    utils.log(f'Saving {path}')
    sitk.WriteImage(image, path)


def load_binvox(path):
    import binvox as bv
    utils.log(f'Loading {path}')
    return bv.Binvox.read(path, mode='dense')


def load_meshio(path):
    import meshio
    utils.log(f'Loading {path}')
    return meshio.read(path)


def save_meshio(path, mesh):
    import meshio
    utils.log(f'Saving {path}')
    meshio.xdmf.write(path, mesh)


def load_trimesh(path, resolver=None):
    import trimesh
    utils.log(f'Loading {path}')
    return trimesh.load_scene(path, resolver=resolver, process=False)


def load_imageio(path, quiet=False):
    import imageio
    if not quiet:
        utils.log(f'Loading {path}')
    return imageio.v2.imread(path)


def load_fenics(path, label_key='label'):
    import fenics as fe
    from mpi4py import MPI
    utils.log(f'Loading {path}')
    mesh, cell_labels = fe.Mesh(), None
    with fe.XDMFFile(MPI.COMM_WORLD, str(mesh_file)) as f:
        f.read(mesh)
        dim = mesh.geometry().dim()
        if label_key:
            cell_labels = fe.MeshFunction('size_t', mesh, dim) # was dolfin
            f.read(cell_labels, label_key)
    return mesh, cell_labels


def load_analyze75(img_file, shape, dtype=None):
    # source: https://stackoverflow.com/questions/27507928/loading-analyze-7-5-format-images-in-python
    utils.log(f'Loading {path}')
    dtype = dtype or np.int16
    array = np.fromfile(img_file, dtype=dtype).reshape(shape)
    item_size = array.dtype.itemsize
    array.strides = (
        item_size,
        item_size * shape[0],
        item_size * shape[0] * shape[1]
    )
    return array.copy()


def load_json(path):
    import json
    utils.log(f'Loading {path}')
    with open(path) as f:
        return json.load(f)


def load_yaml(path):
    import yaml
    utils.log(f'Loading {path}')
    with open(path) as f:
        return yaml.safe_load(f)


def load_xyz(path, dtype=float):
    utils.log(f'Loading {path}')
    with open(path) as f:
        data = [line.strip().split() for line in f]
    return np.array(data, dtype=dtype)

