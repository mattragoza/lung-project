# Physics-constrained learning for lung elasticity estimation from CT images and respiratory deformations

Code for preprocessing lung CT data to generate antomical models and estimate respiratory deformations, train models for lung elasticity estimation using physics-constrained learning, estimate elasticity using physics-based inverse optimization, and evaluate elasticity estimates.

## Environment setup

Run the following commands to create the environment and install the dependencies:

```bash
conda env create -f environment.yml
conda run -n warp pip install "nnunetv2>=2.3.1"
conda run -n warp pip install "TotalSegmentator>=2.5" --no-deps
conda run -n warp pip install "git+ssh://git@github.com/uncbiag/uniGradICONLung.git"
```

## API usage

The main actions are preprocessing, optimization, training, and evalation, which are each run by a script in the `scripts` folder provided with a config file and an optional set of overrides, specified as namespaced keys and values.

Generate COPDGene-derived lung phantom data:

```python
python scripts/preprocess.py config/copdgene.yaml --set NAMESPACE.KEY=VAL
```

Train a physics-constrained learning model:

```python
python scripts/train.py config/copdgene.yaml
```

Run the inverse optimization baseline method:

```python
python scripts/optimize.py config/copdgene.yaml
```

Evaluate the estimated elasticity fields:

```python
python scripts/evaluate.py config/copdgene.yaml
```

## Example usage

Below is python code showing how to generate a list of `Example` objects for a given dataset with a config dict, then access the paths associated with the example.

```python
import project

examples = project.api.get_examples(config={
    'name': 'COPDGene',
    'root': '/restricted/projectnb/batmanlab/mragoza/data/COPDGene',
    'examples': {
        'subjects': ['16514P'],           # list of subject IDs
        'variant': '2026-08-08',          # preprocessing run ID
        'state_pairs': [('EXP', 'INSP')], # list of (fixed_state, moving_state)
        'pipeline_tags': {                # used for constructing file paths
            'image_resampling': 'iso',
            'image_segmentation': 'tsvf',
            'image_registration': 'ugil',
            'anatomical_regions': 'lung',
            'material_properties': 'mat',
            'mesh_generation': 'pyg',
            'mesh_interpolation': 'int',
            'forward_simulation': 'sim'
        }
    }
})
len(examples) # num subjects x num state-pairs
```

Then inspecting `examples[0].paths`:

```python
dict(len=13)
├── 'ref_state':       dict(len=2)
|   ├── 'source_image': PosixPath('/restricted/projectnb/batmanlab/mragoza/data/COPDGene/Images/16514P/Phase-1/RAW/16514P_INSP_STD_TEM_COPD.nii.gz')
|   └── 'source_json':  PosixPath('/restricted/projectnb/batmanlab/mragoza/data/COPDGene/Images/16514P/Phase-1/RAW/16514P_INSP_STD_TEM_COPD.json')
├── 'init_state':      dict(len=5)
|   ├── 'source_image':    PosixPath('/restricted/projectnb/batmanlab/mragoza/data/COPDGene/Images/16514P/Phase-1/RAW/16514P_EXP_STD_TEM_COPD.nii.gz')
|   ├── 'source_json':     PosixPath('/restricted/projectnb/batmanlab/mragoza/data/COPDGene/Images/16514P/Phase-1/RAW/16514P_EXP_STD_TEM_COPD.json')
|   ├── 'resampled_image': PosixPath('/restricted/projectnb/batmanlab/mragoza/data/COPDGene/Processed/2026-08-07/16514P/images/16514P_EXP_iso.nii.gz')
|   ├── 'segment_dir':     PosixPath('/restricted/projectnb/batmanlab/mragoza/data/COPDGene/Processed/2026-08-07/16514P/masks/16514P_EXP_iso_tsvf')
|   └── 'domain_mask':     PosixPath('/restricted/projectnb/batmanlab/mragoza/data/COPDGene/Processed/2026-08-07/16514P/masks/16514P_EXP_iso_tsvf_domain.nii.gz')
├── 'curr_state':      dict(len=5)
|   ├── 'source_image':    PosixPath('/restricted/projectnb/batmanlab/mragoza/data/COPDGene/Images/16514P/Phase-1/RAW/16514P_INSP_STD_TEM_COPD.nii.gz')
|   ├── 'source_json':     PosixPath('/restricted/projectnb/batmanlab/mragoza/data/COPDGene/Images/16514P/Phase-1/RAW/16514P_INSP_STD_TEM_COPD.json')
|   ├── 'resampled_image': PosixPath('/restricted/projectnb/batmanlab/mragoza/data/COPDGene/Processed/2026-08-07/16514P/images/16514P_INSP_iso.nii.gz')
|   ├── 'segment_dir':     PosixPath('/restricted/projectnb/batmanlab/mragoza/data/COPDGene/Processed/2026-08-07/16514P/masks/16514P_INSP_iso_tsvf')
|   └── 'domain_mask':     PosixPath('/restricted/projectnb/batmanlab/mragoza/data/COPDGene/Processed/2026-08-07/16514P/masks/16514P_INSP_iso_tsvf_domain.nii.gz')
├── 'anatomical_map':  PosixPath('/restricted/projectnb/batmanlab/mragoza/data/COPDGene/Processed/2026-08-07/16514P/masks/16514P_EXP_iso_tsvf_lung.nii.gz')
├── 'anatomical_mesh': PosixPath('/restricted/projectnb/batmanlab/mragoza/data/COPDGene/Processed/2026-08-07/16514P/meshes/16514P_EXP_iso_tsvf_lung_pyg.xdmf')
├── 'material_map':    PosixPath('/restricted/projectnb/batmanlab/mragoza/data/COPDGene/Processed/2026-08-07/16514P/fields/16514P_EXP_iso_tsvf_mat_label.nii.gz')
├── 'material_dir':    PosixPath('/restricted/projectnb/batmanlab/mragoza/data/COPDGene/Processed/2026-08-07/16514P/fields/16514P_EXP_iso_tsvf_mat')
├── 'disp_field':      PosixPath('/restricted/projectnb/batmanlab/mragoza/data/COPDGene/Processed/2026-08-07/16514P/fields/16514P_EXP_iso_tsvf_ugil_INSP.nii.gz')
├── 'interp_mesh':     PosixPath('/restricted/projectnb/batmanlab/mragoza/data/COPDGene/Processed/2026-08-07/16514P/meshes/16514P_EXP_iso_tsvf_ugil_INSP_lung_pyg_mat_int.xdmf')
├── 'forward_mesh':    PosixPath('/restricted/projectnb/batmanlab/mragoza/data/COPDGene/Processed/2026-08-07/16514P/meshes/16514P_EXP_iso_tsvf_ugil_INSP_lung_pyg_mat_int_sim.xdmf')
├── 'input_image':     PosixPath('/restricted/projectnb/batmanlab/mragoza/data/COPDGene/Processed/2026-08-07/16514P/images/16514P_EXP_iso.nii.gz')
├── 'domain_mask':     PosixPath('/restricted/projectnb/batmanlab/mragoza/data/COPDGene/Processed/2026-08-07/16514P/masks/16514P_EXP_iso_tsvf_domain.nii.gz')
└── 'target_mesh':     PosixPath('/restricted/projectnb/batmanlab/mragoza/data/COPDGene/Processed/2026-08-07/16514P/meshes/16514P_EXP_iso_tsvf_ugil_INSP_lung_pyg_mat_int_sim.xdmf')
```

The important paths for training are `input_image`, `domain_mask`, and `target_mesh`.

The target mesh should have the following fields defined when loaded with meshio:

```
<meshio mesh object>
  Number of points: 23303
  Number of cells:
    tetra: 99240
  Point data: medit:ref, image, u_reg, E, nu, rho, u_fwd
  Cell data: medit:ref, region, image, u_reg, E, nu, rho, u_fwd
```

## Repository structure

```python
data/
config/             # example configs
    copdgene.yaml
    emory4dct.yaml
    shapenet.yaml
project/            # source code
    core/
    datasets/
    preprocessing/
    training/
    physics/
    visual/
    __init__.py
    api.py
    models.py
    optimization.py
    evaluation.py
    validation.py
    callbacks.py
scripts/            # API runners
    preprocess.py
    validate.py
    optimize.py
    train.py
    evaluate.py
tests/
notebooks/
environment.yml
```

