import sys, os, argparse, pathlib, itertools

import project


def preprocess_visit(visit, variant='ISO', recon='STD', states=['INSP', 'EXP']):

	target = 'EXP'
	for source in states:
		if source == target: continue
		source_name = visit.build_image_name(source, recon)
		target_name = visit.build_image_name(target, recon)
		project.registration.resample_image(visit, variant, )

	# generate segmentation masks and anatomical mesh
	for state in states:
		image_name = visit.build_image_name(state, recon)
		project.registration.resample_images()
		project.segmentation.run_segmentation_tasks(visit, 'ISO', image_name)
		project.segmentation.create_lung_region_mask(visit, 'ISO', image_name)
		project.meshing.generate_anatomical_mesh(visit, 'ISO', image_name)

	# generate deformation field
	for source, target in itertools.product(states, states): 
		if source == target: continue
		source_name = visit.build_image_name(source, recon)
		target_name = visit.build_image_name(target, recon)
		project.deformation.create_deformation_field(visit, variant, source_name, target_name)


@project.utils.main
def preprocess(
	data_file: pathlib.Path,
	data_root: pathlib.Path,
	subject_id: str='all',
	visit_name: str='Phase-1',
	recon: str='STD',
	variant: str='RAW'
):
	ds = project.copdgene.COPDGeneDataset.from_csv(data_file, data_root, visit_name)
	for row, visit in ds:
		if row.sid == subject_id or subject_id == 'all':
			preprocess_visit(visit, variant, recon)

