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

D_CLASH = 1.7
# minimum distance between atoms without clashes

K_SOFTNESS = 3
# if bigger, orbital overlap is more gradual and
# starts at bigger distances
# https://www.desmos.com/calculator/9ignfppxpq

orb_dim_dict = {
                'H Single Bond' : 1,
                'C Single Bond' : 1,
                'O Single Bond' : 1,
                'N Single Bond' : 1,
                'F Single Bond' : 1,
                'Cl Single Bond' : 1,
                'Br Single Bond' : 1,
                'I Single Bond' : 1,

                'C sp' : 1,

                'C sp2' : 1,

                'C sp3' : 1,
                'Br sp3' : 1,

                'O Ether' : 5,
                'S Ether' : 1
                }       
# Half-lenght of the transition state bonding distance involving that atom