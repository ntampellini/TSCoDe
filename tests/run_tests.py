import os
import time
from subprocess import check_call, DEVNULL, STDOUT

os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir(os.path.dirname(os.getcwd()))

import sys
sys.path.append(os.getcwd())

from settings import MOPAC_COMMAND, ORCA_COMMAND, CALCULATOR
from utils import time_to_string


os.chdir('tests')


t_start_run = time.time()


##########################################################################

if CALCULATOR == 'MOPAC':
    check_call(f'{MOPAC_COMMAND} HCOOH.mop > HCOOH.cmdlog 2>&1'.split(), stdout=DEVNULL, stderr=STDOUT)
    
# elif CALCULATOR == 'ORCA':
#     check_call(f'{ORCA_COMMAND} HCOOH.mop > HCOOH.cmdlog 2>&1'.split(), stdout=DEVNULL, stderr=STDOUT)

else:
    print()
    quit()

##########################################################################

try:
    from openbabel import openbabel

except ImportError:
    print(f'ATTENTION: Could not import openbabel python module. Is standalone openbabel correctly installed?')
    quit()

print('\nNo installation faults detected. Running tests.')

os.chdir('tests')
tests = []
for f in os.listdir():
    if f.endswith('.txt'):
        tests.append(os.path.realpath(f))

os.chdir(os.path.dirname(os.getcwd()))
os.chdir(os.path.dirname(os.getcwd()))
# Back to ./TSCoDe

from utils import loadbar
from utils import suppress_stdout_stderr

times = []
for i, f in enumerate(tests):
    name = f.split('\\')[-1].split('/')[-1][:-4] # trying to make it work for both Win and Linux
    loadbar(i, len(tests), f'Running TSCoDe tests ({name}): ')
    
    t_start = time.time()
    with suppress_stdout_stderr():
        check_call(f'python tscode.py {f} {name}'.split(), stdout=DEVNULL, stderr=STDOUT)
                
    t_end = time.time()
    times.append(t_start-t_end)

loadbar(len(tests), len(tests), f'Running TSCoDe tests ({name}): ')    

print()
for i, f in enumerate(tests):
    print('    {:25s}{} s'.format(f.split('\\')[-1].split('/')[-1][:-4], round(times[i], 3)))

print(f'\nTSCoDe tests completed with no errors. ({time_to_string(time.time() - t_start_run)})\n')
