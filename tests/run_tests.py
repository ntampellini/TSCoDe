import os
import sys
import time
from subprocess import CalledProcessError

os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir(os.path.dirname(os.getcwd()))

sys.path.append(os.getcwd())

from settings import COMMANDS, OPENBABEL_OPT_BOOL, CALCULATOR
from optimization_methods import mopac_opt, orca_opt, gaussian_opt, xtb_opt
from utils import HiddenPrints, time_to_string, clean_directory, run_command
from cclib.io import ccread
os.chdir('tests/tests')

t_start_run = time.time()

data = ccread('C2H4.xyz')

print('\nRunning tests for TSCoDe. Settings used:')
print(f'{CALCULATOR=}')

if CALCULATOR == 'MOPAC':
    print(f'MOPAC COMMAND = {COMMANDS[CALCULATOR]}')
    print('\nTesting calculator...')
    mopac_opt(data.atomcoords[0], data.atomnos)

elif CALCULATOR == 'ORCA':
    print(f'ORCA COMMAND = {COMMANDS[CALCULATOR]}')
    print('\nTesting calculator...')
    orca_opt(data.atomcoords[0], data.atomnos)

elif CALCULATOR == 'GAUSSIAN':
    print(f'GAUSSIAN COMMAND = {COMMANDS[CALCULATOR]}')
    print('\nTesting calculator...')
    gaussian_opt(data.atomcoords[0], data.atomnos)

elif CALCULATOR == 'XTB':
    print('\nTesting calculator...')
    xtb_opt(data.atomcoords[0], data.atomnos)

else:
    raise Exception(f'{CALCULATOR} is not a valid calculator. Use MOPAC, ORCA, GAUSSIAN or XTB.')

clean_directory()
print(f'{CALCULATOR} calculator works.')

print(f'\n{OPENBABEL_OPT_BOOL=}')
ff = 'on. Checking its status.' if OPENBABEL_OPT_BOOL else 'off.'
print(f'Force Field optimization is turned {ff}')

if OPENBABEL_OPT_BOOL:
    try:
        print('Trying to import the OpenBabel Python Module...')
        from openbabel import openbabel
        print('Module imported successfully.')

    except ImportError:
        raise Exception(f'Could not import OpenBabel Python module. Is standalone openbabel correctly installed?')

print('\nNo installation faults detected with the current settings. Running tests.')

##########################################################################

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
    name = f.split('\\')[-1].split('/')[-1][:-4] # trying to make it work for either Win, Linux (and Mac?)
    loadbar(i, len(tests), f'Running TSCoDe tests ({name}): ')
    
    t_start = time.time()
    try:
        with HiddenPrints():
            run_command(f'python tscode.py {f} {name}')
            # run_command(f'python tscode.py tests\\tests\\dihedral.txt dihedral')

    except CalledProcessError as error:
        print(error.stderr.decode("utf-8"))
        quit()
                
    t_end = time.time()
    times.append(t_end-t_start)

loadbar(len(tests), len(tests), f'Running TSCoDe tests ({name}): ')    

print()
for i, f in enumerate(tests):
    print('    {:25s}{} s'.format(f.split('\\')[-1].split('/')[-1][:-4], round(times[i], 3)))

print(f'\nTSCoDe tests completed with no errors. ({time_to_string(time.time() - t_start_run)})\n')
