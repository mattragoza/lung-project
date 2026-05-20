from pathlib import Path


class RunOutputs:

    def __init__(self, stage: str, root: str = 'outputs'):
        self.root = Path(root)
        self.stage = str(stage)

    @property
    def base_dir(self):
        return self.root / self.stage

    def csv_path(self, name):
        return self.base_dir / (name + '.csv')

    def mesh_path(self, ex, name):
        return self.base_dir / ex.subject / 'meshes' / (name + '.xdmf')

    def nifti_path(self, ex, name):
        return self.base_dir / ex.subject / 'niftis' / (name + '.nii.gz')

    def raster_dir(self, ex):
        return self.base_dir / ex.subject / 'rasters'

    def raster_path(self, ex, name):
        return self.raster_dir(ex) / (name + '.nii.gz')


