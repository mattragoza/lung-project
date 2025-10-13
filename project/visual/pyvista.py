import nibabel as nib
import pyvista as pv


def view_mesh(mesh):
	p = pv.Plotter()
	p.add_mesh(
	    pv.from_meshio(mesh),
	    scalars='label',
	    cmap='Set1',
	    clim=(0, 8)
	)
	p.show()

