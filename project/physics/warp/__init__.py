import warp as wp
wp.init()
wp.config.quiet = True

from . import solver, forms
