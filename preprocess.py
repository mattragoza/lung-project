import project
import project.preprocessing.api as api

ds = project.datasets.copdgene.COPDGeneDataset(
	data_root='data/COPDGene'
)

examples = ds.examples(
	subjects=['16514P'],
	visits=['Phase-1'],
	variant='ISO',
	state_pairs=[('EXP', 'INSP')],
	recon='STD',
	mask_name='lung_regions',
	mesh_tag='volume'
)

for ex in examples:

	api.resample_image_on_reference(
		input_path=ex.paths['fixed_source'],
		output_path=ex.paths['fixed_image'],
		ref_path=ex.paths['ref_image']
	)
	api.resample_image_using_reference(
		input_path=ex.paths['moving_source'],
		output_path=ex.paths['moving_image'],
		ref_path=ex.paths['ref_image']
	)
	api.create_segmentation_masks(
		image_path=ex.paths['fixed_image'],
		output_dir=ex.paths['fixed_mask'].parent
	)
	api.create_multi_region_mask(
		mask_dir=ex.paths['fixed_mask'].parent,
		output_path=ex.paths['fixed_mask']
	)
	api.create_anatomical_meshes(
		mask_path=ex.paths['fixed_mesh'],
		output_path=ex.paths['fixed_mesh'],
	)
	api.create_corrfield_displacement(
		fixed_path=ex.paths['fixed_image'],
		moving_path=ex.paths['moving_image'],
		mask_path=ex.paths['fixed_mask'],
		output_path=ex.paths['disp_field']
	)

