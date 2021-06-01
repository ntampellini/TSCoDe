import os
from parameters import MOPAC_COMMAND
from subprocess import check_output

os.chdir(os.path.dirname(os.path.realpath(__file__)))

##########################################################################

check_output(f'{MOPAC_COMMAND} HCOOH.mop > HCOOH.cmdlog 2>&1', shell=True)
    
##########################################################################

try:
    from openbabel import openbabel

except ImportError:
    print(f'ATTENTION: Could not import openbabel python module. Is standalone openbabel correctly installed?')

print('\nTSCoDe tests completed with no errors.\n')
