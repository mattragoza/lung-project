# preprocessing/stages/synthetic_images.py

from typing import List, Dict, Tuple, Any
from pathlib import Path

from ...core import utils, fileio


def generate_image(mask_path, output_path, config, random_seed=0):
    utils.check_keys(
        config,
        valid={'material_catalog', 'texture_source', 'intensity_model', 'noise_model', 'use_simple'},
        where='image_generation'
    )
    from .. import materials, textures, image_synthesis

    nifti = fileio.load_nibabel(mask_path)
    mask = nifti.get_fdata().astype(int)

    mat_df = materials.load_material_catalog(config['material_catalog'])

    tex_path = config['texture_source']['annotations']
    tex_df = textures.load_texture_annotations(tex_path)

    use_solid = config['texture_source']['use_solid']
    tex_cache = textures.TextureCache(tex_df)

    proc_kws = config['texture_source']['preprocessing']
    proc_spec = textures.PreprocessSpec(**proc_kws)

    def texture_map(label: int):
        tid = mat_df.loc[label].texture_id
        return tex_cache.get(tid, use_solid, proc_spec)

    utils.log('Computing intensity model')
    intensity_kws = config.get('intensity_model', {})
    outputs = materials.compute_intensity_model(
        mat_df['density_val'], mat_df['elastic_val'], **intensity_kws
    )
    mat_df['density_feat'] = outputs['density_feat']
    mat_df['elastic_feat'] = outputs['elastic_feat']
    mat_df['intensity_bias'] = outputs['intensity_bias']
    mat_df['intensity_range'] = outputs['intensity_range']
    utils.log(mat_df)

    utils.log('Generating volumetric image')
    if config.get('use_simple', False):
        rgb = not proc_spec.grayscale
        image = image_synthesis.generate_simple_image(
            mask, texture_map, seed=random_seed, rgb=rgb
        )
    else:
        noise_kws = config.get('noise_model', {})
        image = image_synthesis.generate_volumetric_image(
            mask, nifti.affine, mat_df, tex_cache, **noise_kws, random_seed=random_seed
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fileio.save_nibabel(output_path, image, nifti.affine)

