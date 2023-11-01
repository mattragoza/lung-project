# Lung biomechanical modeling project

## Emory 4DCT data download

1. Go to [download page](https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/downloads-and-reference-data/4dct.html)
2. Submit [access request form](https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/access-request-form.html)
	- Landing page contains password
3. For each case packet i=1..10,
	- Follow the download link (DropBox)
	- Enter password and download `Case${i}Pack.zip`
	- Move .zip file to `lung-project/data/download`
4. Unzip case packets into `lung-project/data/Emory-4DCT`
	- Use the commands below:

```bash
cd lung-project
for i in {1..10};
	do unzip data/download/Case${i}Pack.zip -d data/Emory-4DCT;
done
```
## Conda environment setep

Run the following to create the conda environment and register it as jupyter notebook kernel:

```bash
mamba env create --file=environment.yml
mamba activate 4DCT
python -m ipykernel install --user --name=4DCT
```

## SimpleElastix installation

We have to build this from source code, but it is pretty easy. Follow the commands below:

https://simpleelastix.readthedocs.io/GettingStarted.html

```bash
# build from source code (takes ~2 hours)
git clone https://github.com/SuperElastix/SimpleElastix
mkdir SimpleElastix/build
cd SimpleElastix/build
cmake ../SuperBuild/ -DWRAP_DEFAULT=OFF -DWRAP_PYTHON=ON
make -j8

# install python bindings (fast)
cd SimpleITK-build/Wrapping/Python
mamba activate 4DCT
python Packaging/setup.py install --prefix=$CONDA_PREFIX
```

