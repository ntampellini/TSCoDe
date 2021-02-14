# Scalar fields constants
VOXEL_DIM = 0.1

# Sphere stamps constants
STAMP_SIZE = 2
HARDNESS = 0.5

# Distribution-correcting constants
BREADTH = 2

##############################################

# Monte Carlo search constants

MAX_ANGLE = 20
# Maximum angle, in degrees, at which initial
# population and sequential steps can be
# rotated along x, y, or z axis

MAX_DIST = 0.5
# Maximum traslation, in angstroms, at which initial
# population and sequential steps can be
# rotated along x, y, or z axis

D_EQ = 1.7
# ideal distance between reactive atoms

K_SOFTNESS = 3
# if bigger, orbital overlap is more gradual and
# starts at bigger distances
# https://www.desmos.com/calculator/9ignfppxpq
