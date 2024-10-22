import pathlib
import numpy as np
import xarray as xr
import nibabel as nib
import pygalmesh

from . import meshing, interpolation, pde


class PhantomSet(object):
    
    def __init__(self, data_root, num_phantoms):
        self.data_root = pathlib.Path(data_root)
        self.phantoms = []
        for i in range(num_phantoms):
            phantom = Phantom(data_root, phantom_id=i)
            self.phantoms.append(phantom)
            
    def generate(self, *args, **kwargs):
        self.data_root.mkdir(exist_ok=True)
        for i, phantom in enumerate(self.phantoms):
            phantom.generate(*args, **kwargs)
            
    def get_examples(self, mesh_radius):
        examples = []
        for i, phantom in enumerate(self.phantoms):
            examples.append({
                'name': phantom.phantom_name,
                'anat_file': phantom.anat_file(),
                'disp_file': phantom.disp_file(),
                'mask_file': phantom.mask_file(),
                'mesh_file': phantom.mesh_file(mesh_radius),
                'mesh_radius': mesh_radius
            })
        return examples


class Phantom(object):
    
    def __init__(self, data_root, phantom_id):
        self.data_root = pathlib.Path(data_root)
        self.phantom_id = int(phantom_id)
        
    @property
    def phantom_name(self):
        return f'phantom{self.phantom_id}'
        
    @property
    def phantom_dir(self):
        return self.data_root / self.phantom_name
        
    def anat_file(self):
        return self.phantom_dir / f'{self.phantom_name}_anat.nii.gz'

    def disp_file(self):
        return self.phantom_dir / f'{self.phantom_name}_disp.nii.gz'
    
    def elast_file(self):
        return self.phantom_dir / f'{self.phantom_name}_elast.nii.gz'
    
    def mask_file(self):
        return self.phantom_dir / f'{self.phantom_name}_mask.nii.gz'
    
    def mesh_file(self, mesh_radius):
        return self.phantom_dir / f'{self.phantom_name}_mesh{mesh_radius}.xdmf'
    
    def generate(self, shape, resolution, mesh_radius, **kwargs):
        self.phantom_dir.mkdir(exist_ok=True)
        elast, anat, disp_bc, disp, mask, mesh = generate_phantom(
            random_seed=self.phantom_id,
            shape=shape,
            resolution=resolution,
            mesh_radius=mesh_radius,
            mesh_file=self.mesh_file(mesh_radius),
            **kwargs
        )
        self.elast = project.utils.as_xarray(
            elast, dims=['x', 'y', 'z'], name='elasticity'
        )

        x = np.arange(shape[0]) * resolution[0]
        y = np.arange(shape[1]) * resolution[1]
        z = np.arange(shape[2]) * resolution[2]

        self.elast = xr.DataArray(
            data=elast,
            dims=['x', 'y', 'z'],
            coords=dict(x=x, y=y, z=z),
            name='elasticity'
        )
        self.anat = xr.DataArray(
            data=anat,
            dims=['x', 'y', 'z'],
            coords=dict(x=x, y=y, z=z),
            name='anatomy'
        )
        self.disp = xr.DataArray(
            data=disp,
            dims=['x', 'y', 'z', 'component'],
            coords=dict(x=x, y=y, z=z, component=['x', 'y', 'z']),
            name='displacement'
        )
        self.mask = xr.DataArray(
            data=mask,
            dims=['x', 'y', 'z'],
            coords=dict(x=x, y=y, z=z),
            name='mask'
        )
        affine = np.diag(list(resolution) + [1])
        
        print(f'Saving {self.elast_file()}')
        nib.save(nib.Nifti1Image(self.elast.data, affine), self.elast_file())

        print(f'Saving {self.anat_file()}')
        nib.save(nib.Nifti1Image(self.anat.data, affine), self.anat_file())
              
        print(f'Saving {self.disp_file()}')
        nib.save(nib.Nifti1Image(self.disp.data, affine), self.disp_file())
              
        print(f'Saving {self.mask_file()}')
        nib.save(nib.Nifti1Image(self.mask.data, affine), self.mask_file())  



def spatial_coordinates(shape, resolution):
    x = np.arange(shape[0]) * resolution[0]
    y = np.arange(shape[1]) * resolution[1]
    z = np.arange(shape[2]) * resolution[2]
    return np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)


def frequency_coordinates(shape):
    x = np.fft.fftfreq(shape[0])
    y = np.fft.fftfreq(shape[1])
    z = np.fft.fftfreq(shape[2])
    return np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)


def draw_target(coords, center, radius, type_):
    assert type_ in {'disk', 'sphere', 'cube'}
    if type_ == 'disk':
        component = (0,1)
        order = 2
    elif type_ == 'sphere':
        component = (0,1,2)
        order = 2
    elif type_ == 'cube':
        component = (0,1,2)
        order = np.inf
    displacement = (coords - center)[...,component]
    distance = np.linalg.norm(displacement, ord=order, axis=-1)
    return (distance < radius).astype(int)


def butterworth_filter(freqs, cutoff, power):
    radius = np.linalg.norm(freqs, axis=-1)
    return 1 / (1 + (radius/cutoff)**power)


def bandpass_filter(freqs, cutoff1, cutoff2, power):
    filter1 = butterworth_filter(freqs, cutoff1, power)
    filter2 = butterworth_filter(freqs, cutoff2, power)
    return (1 - filter1) * filter2


def random_texture(filter):
    complex_noise = (
        np.random.normal(0, 1, filter.shape) + 
        np.random.normal(0, 1, filter.shape) * 1j
    )
    texture = np.fft.ifftn(complex_noise * filter).real
    texture -= texture.min()
    texture /= texture.max()
    return texture * 2 - 1


def generate_phantom(
    shape=(256, 256, 64),
    resolution=(1.0, 1.0, 2.0),
    target_type='disk',
    target_radius=50,
    min_log_kpa=-1,
    max_log_kpa=1,
    bias_midpoint=0,
    bias_range=500,
    anat_range=2000,
    log_cutoff_min=-3,
    log_cutoff_max=0,
    phase_sigma=1.0,
    mesh_radius=10,
    mesh_file='phantom.xdmf'
):
    # define coordinates over spatial domain
    print('Defining spatial domain...')
    coords = spatial_coordinates(shape, resolution)
    center = coords.mean(axis=(0,1,2), keepdims=True)

    # define spatial regions for target and background
    target = draw_target(coords, center, target_radius, target_type)
    region0 = (target == 0).astype(np.uint16)
    region1 = (target == 1).astype(np.uint16)

    # sample latent variables for each region
    print('Sampling latent variables...')
    latent0 = np.random.rand()
    latent1 = np.random.rand()
    print(latent0, latent1)

    # map latent variables to stiffness values
    print('Generating stiffness map..')
    log_kpa_range = (max_log_kpa - min_log_kpa)
    log_kpa0 = latent0 * log_kpa_range + min_log_kpa
    log_kpa1 = latent1 * log_kpa_range + min_log_kpa

    # assign stiffness to each spatial region
    mu0 = 10**log_kpa0 * 1000
    mu1 = 10**log_kpa1 * 1000
    mu = region0 * mu0 + region1 * mu1

    # map latent variables to anatomical texture and bias
    print('Generating anatomical image...')
    bias_min = bias_midpoint - bias_range/2
    bias0 = latent0 * bias_range + bias_min
    bias1 = latent1 * bias_range + bias_min

    log_cutoff_range = (log_cutoff_max - log_cutoff_min)
    log_cutoff0 = latent0 * log_cutoff_range + log_cutoff_min
    log_cutoff1 = latent1 * log_cutoff_range + log_cutoff_min

    cutoff0 = 10**log_cutoff0
    cutoff1 = 10**log_cutoff1

    filter0 = bandpass_filter(freqs, cutoff0/2, cutoff0, power=2)
    filter1 = bandpass_filter(freqs, cutoff1/2, cutoff1, power=2)

    texture0 = random_texture(filter0)
    texture1 = random_texture(filter1)
    texture_range = (anat_range - bias_range)

    anat0 = texture0 * texture_range/2 + bias0
    anat1 = texture1 * texture_range/2 + bias1
    anat = region0 * anat0 + region1 * anat1
    
    # generate random displacement boundary condition
    phase = np.random.normal(0, phase_sigma, (3, 3))
    extent = np.max(shape) * np.array(resolution)
    disp_bc = np.sin(2 * np.pi * (coords/extent) @ phase)

    # generate mesh using pygalmesh
    mask = (target + 1).astype(np.uint16)
    print('Generating mesh...')
    mesh = pygalmesh.generate_from_array(
        mask, resolution,
        max_cell_circumradius=float(mesh_radius),
        odt=True
    )
    mesh = project.meshing.remove_unused_points(mesh)
    
    # save mesh using meshio, then read with fenics
    mesh_cells = [(mesh.cells[1].type, mesh.cells[1].data)]
    meshio.write_points_cells(mesh_file, mesh.points, mesh_cells)
    mesh = project.data.load_mesh_file(mesh_file)
    
    # convert to FEM basis coefficients
    device = 'cpu'
    dtype = torch.float32
    pde = project.pde.LinearElasticPDE(mesh)
    print('Interpolating FEM coefficients...')
    u_dofs = project.interpolation.image_to_dofs(
        torch.as_tensor(disp_bc, dtype=dtype, device=device).permute(3, 0, 1, 2),
        resolution, pde.V,
        radius=int(mesh_radius),
        sigma=mesh_radius/2
    )
    mu_dofs = project.interpolation.image_to_dofs(
        torch.as_tensor(mu, dtype=dtype, device=device),
        resolution, pde.S,
        radius=int(mesh_radius),
        sigma=mesh_radius/2
    )
    anat_dofs = project.interpolation.image_to_dofs(
        torch.as_tensor(anat, dtype=dtype, device=device),
        resolution, pde.S,
        radius=int(mesh_radius),
        sigma=mesh_radius/2
    )
    rho_dofs = (1 + anat_dofs/1000) * 1000
    
    # solve FEM for simulated displacement dofs
    print('Solving FEM model...')
    u_sim_dofs = pde.forward(
        u_dofs.unsqueeze(0),
        mu_dofs.unsqueeze(0),
        rho_dofs.unsqueeze(0),
    )[0]
    
    # convert to displacement image
    print('Converting to image...')
    disp_sim = project.interpolation.dofs_to_image(
        u_sim_dofs, pde.V, disp_bc.shape[:3], resolution
    ).permute(1,2,3,0)
    
    return mu, anat, disp_bc, disp_sim
