# physics/warp/__init__.py

import warp as wp

wp.init()

#wp.config.quiet = True
#wp.config.debug = True

from .solver import WarpFEMSolver
