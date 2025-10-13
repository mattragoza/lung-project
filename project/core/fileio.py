from pathlib import Path
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import meshio
import yaml


def load_image(path: str|Path, using='nibabel'):
	if using == 'nibabel':
		return nib.load(path)
	elif using == 'simpleitk':
		return sitk.ReadImage(path)
	raise RuntimeError(using)


def load_mesh(path: str|Path, using='meshio'):
	if using == 'meshio':
		return meshio.read(path)
	elif using == 'fenics':
		return load_mesh_with_fenics(path)
	raise RuntimeError(using)


def load_mesh_with_fenics(mesh_file, label_key='label'):
    import fenics as fe
    from mpi4py import MPI
    import dolfin
    mesh = fe.Mesh()
    with fe.XDMFFile(MPI.COMM_WORLD, str(mesh_file)) as f:
        f.read(mesh)
        dim = mesh.geometry().dim()
        if label_key:
            cell_labels = dolfin.MeshFunction('size_t', mesh, dim)
            f.read(cell_labels, label_key)
        else:
            cell_labels = None
    return mesh, cell_labels


def load_yaml_file(yaml_file):
    '''
    Read a YAML configuration file.
    '''
    print(f'Loading {yaml_file}')
    with open(yaml_file) as f:
        return yaml.safe_load(f)


def load_xyz_file(xyz_file, dtype=float):
    '''
    Read landmark xyz coordinates from text file.
    '''
    print(f'Loading {xyz_file}')
    with open(xyz_file) as f:
        data = [line.strip().split() for line in f]
    return np.array(data, dtype=dtype)


def load_img_file(img_file, shape, dtype=np.int16):
    '''
    Read CT image from file in Analyze 7.5 format.
    
    https://stackoverflow.com/questions/27507928/loading-analyze-7-5-format-images-in-python
    '''
    array = np.fromfile(img_file, dtype).reshape(shape)
    item_size = array.dtype.itemsize
    array.strides = (
        item_size,
        item_size * shape[0],
        item_size * shape[0] * shape[1]
    )
    return array.copy()
