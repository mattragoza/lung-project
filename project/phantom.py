import pathlib
import numpy as np
import xarray as xr
import nibabel as nib
import scipy.stats
import pygalmesh
import meshio
import torch

from . import data, meshing, interpolation, pde, utils


class PhantomSet(object):
    
    def __init__(self, data_root, phantom_ids):
        self.data_root = pathlib.Path(data_root)
        self.phantoms = []
        for i in phantom_ids:
            phantom = Phantom(data_root, phantom_id=i)
            self.phantoms.append(phantom)

    def __len__(self):
        return len(self.phantoms)

    def __getitem__(self, idx):
        return self.phantoms[idx]
            
    def generate(self, *args, **kwargs):
        self.data_root.mkdir(exist_ok=True)
        for i, phantom in enumerate(self.phantoms):
            phantom.generate(*args, **kwargs)
            
    def get_examples(self, mesh_version):
        examples = []
        for i, phantom in enumerate(self.phantoms):
            examples.append({
                'name': phantom.phantom_name,
                'anat_file': phantom.anat_file(),
                'elast_file': phantom.elast_file(),
                'disp_file': phantom.disp_file(),
                'mask_file': phantom.mask_file(),
                'mesh_file': phantom.mesh_file(mesh_version),
                'has_labels': (mesh_version >= 20)
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
    
    def mesh_file(self, mesh_version):
        return self.phantom_dir / f'{self.phantom_name}_mesh{mesh_version}.xdmf'
    
    def load_niftis(self):

        print(f'Loading {self.anat_file()}')
        nifti = nib.load(self.anat_file())
        shape = nifti.header.get_data_shape()
        resolution = nifti.header.get_zooms()

        self.shape = shape
        self.resolution = resolution

        x = np.arange(shape[0]) * resolution[0]
        y = np.arange(shape[1]) * resolution[1]
        z = np.arange(shape[2]) * resolution[2]

        self.anat = xr.DataArray(
            data=nifti.get_fdata(),
            dims=['x', 'y', 'z'],
            coords=dict(x=x, y=y, z=z),
            name='CT'
        )
        print(f'Loading {self.elast_file()}')
        nifti = nib.load(self.elast_file())
        self.elast = xr.DataArray(
            data=nifti.get_fdata(),
            dims=['x', 'y', 'z'],
            coords=dict(x=x, y=y, z=z),
            name='elasticity'
        )
        print(f'Loading {self.disp_file()}')
        nifti = nib.load(self.disp_file())
        self.disp = xr.DataArray(
            data=nifti.get_fdata(),
            dims=['x', 'y', 'z', 'component'],
            coords=dict(x=x, y=y, z=z, component=['x', 'y', 'z']),
            name='displacement'
        )
        print(f'Loading {self.mask_file()}')
        nifti = nib.load(self.mask_file())
        self.mask = xr.DataArray(
            data=nifti.get_fdata(),
            dims=['x', 'y', 'z'],
            coords=dict(x=x, y=y, z=z),
            name='mask'
        )

    def load_mesh(self, mesh_version):
        mesh_file = self.mesh_file(mesh_version)
        print(f'Loading {mesh_file}')
        has_labels = (mesh_version >= 20)
        self.mesh, self.cell_labels = meshing.load_mesh_fenics(mesh_file, has_labels)

    def generate(self, mask_file, mesh_version, **kwargs):
        self.phantom_dir.mkdir(exist_ok=True)
        
        nifti = nib.load(mask_file)
        shape = nifti.header.get_data_shape()
        resolution = nifti.header.get_zooms()
        input_mask = (nifti.get_fdata() > 0).astype(int)
        
        elast, anat, disp_bc, disp, mask, mesh = generate_phantom(
            input_mask=input_mask,
            resolution=resolution,
            mesh_file=self.mesh_file(mesh_version),
            random_seed=self.phantom_id,
            **kwargs 
        )

        # convert to xarrays
        x = np.arange(shape[0]) * resolution[0]
        y = np.arange(shape[1]) * resolution[1]
        z = np.arange(shape[2]) * resolution[2]

        self.elast = utils.as_xarray(
            elast,
            dims=['x', 'y', 'z'],
            coords=dict(x=x, y=y, z=z),
            name='elasticity'
        )
        self.anat = utils.as_xarray(
            anat,
            dims=['x', 'y', 'z'],
            coords=dict(x=x, y=y, z=z),
            name='anatomy'
        )
        self.disp = utils.as_xarray(
            disp,
            dims=['x', 'y', 'z', 'component'],
            coords=dict(x=x, y=y, z=z, component=['x', 'y', 'z']),
            name='displacement'
        )
        self.mask = utils.as_xarray(
            mask,
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


def draw_target(coords, center, radius, rotation):
    displacement = ((coords - center) @ rotation) / radius
    distance = np.linalg.norm(displacement, ord=2, axis=-1)
    return (distance <= 1).astype(int)


def draw_random_target(coords, radius_min, radius_max, pad=0.2, iso=False):
    coords_min = coords.min(axis=(0,1,2))
    coords_max = coords.max(axis=(0,1,2))
    coords_range = coords_max - coords_min
    center_min = coords_min + coords_range * pad
    center_max = coords_max - coords_range * pad
    center = np.random.uniform(center_min, center_max)
    radius = np.random.uniform(radius_min, radius_max, (1 if iso else 3))
    rotation = scipy.stats.ortho_group.rvs(3)
    return draw_target(coords, center, radius, rotation)


def add_non_overlapping_targets(
    regions, resolution, max_targets, radius_min, radius_max
):
    coords = spatial_coordinates(regions.shape, resolution)
    regions = np.array(regions)
    target_count = 0
    attempt_count = 0
    while (target_count < max_targets) and (attempt_count < max_targets * 10):
        target = draw_random_target(coords, radius_min, radius_max)
        collisions = np.logical_and(target, (regions != 1))
        if not np.any(collisions):
            target_count += 1
            regions += target * target_count
        attempt_count += 1
    print(target_count, attempt_count)
    return regions


def butterworth_filter(freqs, cutoff, power):
    radius = np.linalg.norm(freqs, axis=-1)
    return 1 / (1 + (radius/cutoff)**power)


def bandpass_filter(freqs, cutoff1, cutoff2, power):
    filter1 = butterworth_filter(freqs, cutoff1, power)
    filter2 = butterworth_filter(freqs, cutoff2, power)
    return (1 - filter1) * filter2


def draw_random_texture(shape, cutoff1, cutoff2, power):
    freq_coords = frequency_coordinates(shape)
    filter_ = bandpass_filter(freq_coords, cutoff1, cutoff2, power)
    complex_noise = (
        np.random.normal(0, 1, shape) + 
        np.random.normal(0, 1, shape) * 1j
    )
    texture = np.fft.ifftn(complex_noise * filter_).real
    texture -= texture.min()
    texture /= texture.max()
    return texture * 2 - 1


def generate_phantom(
    input_mask,
    resolution,
    max_targets=5,
    radius_min=5,
    radius_max=25,
    log_kpa_min=-1,
    log_kpa_max=2,
    bias_midpoint=-750,
    bias_range=250,
    anat_range=500,
    log_cutoff_min=-3,
    log_cutoff_max=0,
    phase_sigma=1.0,
    mesh_file='phantom.xdmf',
    interp_size=5,
    interp_type='tent',
    rho_value=0,
    dummy_targets=False,
    random_seed=None
):
    print(f'Setting random seed to {random_seed}')
    np.random.seed(random_seed)
    
    # define coordinates over spatial domain
    print('Defining spatial domain...')
    shape = input_mask.shape
    coords = spatial_coordinates(shape, resolution)

    # define spatial regions for target and background
    regions = add_non_overlapping_targets(
        input_mask, resolution, max_targets, radius_min, radius_max
    )
    n_regions = 1 + int(regions.max())
    region_id = np.arange(n_regions)
    region_indicator = (regions[None,...] == region_id[:,None,None,None])

    # sample latent variables for each region
    print('Sampling latent variables...')
    latent = np.random.rand(n_regions)

    # determine which regions are dummmy regions
    dummy = (region_id > 1) & (region_id % 2 == 1) & dummy_targets

    # map latent variables to stiffness values
    print('Generating stiffness map..')
    log_kpa_range = (log_kpa_max - log_kpa_min)
    log_kpa = latent * log_kpa_range + log_kpa_min
    elast = np.power(10, log_kpa) * 1000 # log kPa -> Pa
    elast[0] = 0 # background stiffness
    elast[dummy] = elast[1] # dummy stiffness

    # assign stiffness to each spatial region
    elast = (region_indicator * elast[:,None,None,None]).sum(axis=0)

    # map latent variables to anatomical texture parameters
    print('Generating anatomical image...')
    bias_min = bias_midpoint - bias_range/2
    bias = latent * bias_range + bias_min

    log_cutoff_range = (log_cutoff_max - log_cutoff_min)
    log_cutoff = latent * log_cutoff_range + log_cutoff_min
    cutoff = np.power(10, log_cutoff)

    texture = np.zeros_like(region_indicator, dtype=float)
    for i in range(n_regions):
        if dummy[i]:
            texture[i] = texture[1]
        else:
            texture[i] = draw_random_texture(shape, cutoff[i]/2, cutoff[i], power=3)

    texture_range = (anat_range - bias_range)
    anat = texture * texture_range/2 + bias[:,None,None,None]
    anat_min = bias_midpoint - anat_range/2
    anat[0] = anat_min # background texture

    # assign anatomical textures to each region
    anat = (region_indicator * anat).sum(axis=0)
    
    # generate random displacement boundary condition
    print('Generating displacement BC...')
    phase = np.random.normal(0, phase_sigma, (3, 3))
    extent = np.max(shape) * np.array(resolution)
    disp_bc = np.sin(2 * np.pi * (coords/extent) @ phase)

    # generate mesh using pygalmesh
    print('Generating mesh...', flush=True)
    regions = regions.astype(np.uint16)
    mesh = pygalmesh.generate_from_array(
        regions, resolution,
        max_cell_circumradius=10.0,
        min_facet_angle=15,
        max_facet_distance=1.0,
        odt=True, lloyd=True
    )
    mesh = meshing.remove_unused_points(mesh)
    
    # save mesh using meshio, then read with fenics
    meshing.save_mesh_meshio(mesh_file, mesh, cell_blocks=[1])
    mesh = meshing.load_mesh_fenics(mesh_file)
    mask = (regions > 0)
    
    # convert to FEM basis coefficients
    print('Interpolating FEM coefficients...')
    fem = pde.FiniteElementModel(mesh, resolution)

    anat = torch.as_tensor(anat, dtype=torch.float32, device='cuda')
    elast = torch.as_tensor(elast, dtype=torch.float32, device='cuda')
    disp_bc = torch.as_tensor(disp_bc, dtype=torch.float32,device='cuda')
    mask = torch.as_tensor(mask, dtype=torch.float32, device='cuda')

    points = fem.points.to('cuda')
    radius = fem.radius.to('cuda')

    u_dofs = interpolation.interpolate_image(
        disp_bc.permute(3, 0, 1, 2), mask.unsqueeze(0), resolution, points, radius,
        kernel_size=interp_size,
        kernel_type=interp_type,
    ).to(dtype=torch.float64, device='cpu')

    e_dofs = interpolation.interpolate_image(
        elast.unsqueeze(0), mask.unsqueeze(0), resolution, points, radius,
        kernel_size=interp_size,
        kernel_type=interp_type,
    ).to(dtype=torch.float64, device='cpu')

    a_dofs = interpolation.interpolate_image(
        anat.unsqueeze(0), mask.unsqueeze(0), resolution
        , points, radius,
        kernel_size=interp_size,
        kernel_type=interp_type,
    ).to(dtype=torch.float64, device='cpu')

    if rho_value == 'anat':
        rho_dofs = (a_dofs + 1000)
    else:
        rho_dofs = torch.full_like(a_dofs, float(rho_value))
    
    # solve for simulated displacement dofs
    print('Solving FEM model...')
    u_sim_dofs = fem.forward(
        u_dofs[None,:,:],
        e_dofs[None,:,0],
        rho_dofs[None,:,0],
    )[0]
    
    # convert to displacement image
    print('Converting displacement field to image...')
    disp_sim = interpolation.dofs_to_image(
        u_sim_dofs, fem.V, disp_bc.shape[:3], resolution
    ).permute(1,2,3,0)
    
    print(anat.shape, elast.shape, disp_bc.shape, disp_sim.shape, regions.shape)

    return elast, anat, disp_bc, disp_sim, regions, mesh
