import pandas as pd


DENSITY_COLS = ['density_id', 'density_kgm3', 'frequency']
DENSITY_DATA = [
	('Dense', 500., 0.75),
	('Porous', 50., 0.25),
]
ELASTIC_COLS = ['elastic_id', 'elastic_kpa', 'frequency']
ELASTIC_DATA = [
	('Soft',   1.0, 0.25),
	('Medium', 3.0, 0.50),
	('Hard',   9.0, 0.25),
]

def build_material_catalog(density_data=None, elastic_data=None):

	dens = pd.DataFrame(density_data or DENSITY_DATA, columns=DENSITY_COLS)
	elas = pd.DataFrame(elastic_data or ELASTIC_DATA, columns=ELASTIC_COLS)

	mats = pd.merge(dens, elas, how='cross')
	mats['material_key'] = mats['density_key'] + mats['elastic_key']
	mats['frequency'] = mats['density_freq'] * mats['elastic_freq']

	return mats.sort_values('material_key', ignore_index=True)

