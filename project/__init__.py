from . import (

	# updated modules
	copdgene,
	registration,
	segmentation,
	meshing,
	interpolation,
	transforms,

	# probably in fine shape
	data,       # torch data loader
	model,      # torch nn modules
	training,   # trainer class
	evaluation, # evaluator class + metrics
	output,     # interactive training plot
	utils,

	# old data sets, update or remove?
	emory4dct,
	phantom,

	pde, # fenics stuff. update or remove?
	visual # does this even belong in this package?
)
