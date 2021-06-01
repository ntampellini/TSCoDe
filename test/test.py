import os
from parameters import *
from optimization_methods import suppress_stdout_stderr
os.chdir(os.path.dirname(os.path.realpath(__file__)))

##########################################################################

with suppress_stdout_stderr():
    # mopac_exit_code = os.system(MOPAC_COMMAND + ' HCOOH.mop')
    mopac_exit_code = os.system('mopacdd' + ' HCOOH.mop')

if mopac_exit_code == 0:
    pass

elif mopac_exit_code == 1:
    print(f'ATTENTION: Command \'{MOPAC_COMMAND}\' is not recognized by the prompt (Error code 1). Did you add the MOPAC folder to PATH?')

else:
    print(f'ATTENTION: Command \'{MOPAC_COMMAND}\' had non-zero exit code {mopac_exit_code}.')

##########################################################################

try:
    from openbabel import openbabel
except:
    print(f'ATTENTION: Could not import openbabel python module. Is standalone openbabel correctly installed?')
