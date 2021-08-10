'''

TSCODE: Transition State Conformational Docker
Copyright (C) 2021 Nicolò Tampellini

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

'''

def run_tests():
    
    import os
    import time
    from subprocess import CalledProcessError

    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    # sys.path.append(os.getcwd())

    from settings import COMMANDS, OPENBABEL_OPT_BOOL, CALCULATOR
    from optimization_methods import mopac_opt, orca_opt, gaussian_opt, xtb_opt
    from utils import HiddenPrints, time_to_string, clean_directory, run_command, loadbar
    from cclib.io import ccread

    os.chdir(os.path.dirname(os.getcwd()))
    os.chdir('tests')

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

    # os.chdir(os.path.dirname(os.getcwd()))
    # os.chdir('tscode')
    # # Back to ./tscode

    times = []
    for i, f in enumerate(tests):
        name = f.split('\\')[-1].split('/')[-1][:-4] # trying to make it work for either Win, Linux (and Mac?)
        loadbar(i, len(tests), f'Running TSCoDe tests ({name}): ')
        
        t_start = time.time()
        try:
            with HiddenPrints():
                run_command(f'python -m tscode {f} -n {name}')

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
