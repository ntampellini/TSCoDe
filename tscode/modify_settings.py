# coding=utf-8
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

def run_setup():
    '''
    Invoked by the command
    > python -m tscode -setup

    Guides the user in setting up the calculation options
    contained in the settings.py file.
    '''

    import os
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    from settings import DEFAULT_LEVELS, COMMANDS

    properties = {
        'OPENBABEL_OPT_BOOL':False,
        'CALCULATOR':None,
        'NEW_DEFAULT':None,
        'NEW_COMMAND':None,
        'PROCS':1,
        'MEM_GB':1,
    }

    def ask(text, accepted=('y','n'), default='n'):
        text = '--> ' + text
        if accepted is None:
            return input(text)
        answer = None
        while answer not in accepted:
            answer = input(text)
            answer = answer if answer != '' else default
        return answer

    print('\nTSCoDe setup:\n')

    #########################################################################################

    answer = ask('Would you like to use the OpenBabel Force Field implementation? y/[n]: ')
    if answer == 'y':
        properties['OPENBABEL_OPT_BOOL'] = True

    #########################################################################################

    answer = ask('What calculator would like to use?\n- MOPAC -> m\n- ORCA -> o\n- Gaussian -> g\n- XTB -> x\n\nAnswer m/o/g/[x]: ',
                accepted=('m','o','g','x'), default='x')

    if answer == 'm':
        properties['CALCULATOR'] = 'MOPAC'

    elif answer == 'o':
        properties['CALCULATOR'] = 'ORCA'

    elif answer == 'g':
        properties['CALCULATOR'] = 'GAUSSIAN'

    elif answer == 'x':
        properties['CALCULATOR'] = 'XTB'

    #########################################################################################

    answer = ask((f'The default level for {properties["CALCULATOR"]} calculations is {DEFAULT_LEVELS[properties["CALCULATOR"]]}. ' +
                'If you would like to change it, type it here, otherwise press enter : '), accepted=None)
    if answer != '':
        properties['NEW_DEFAULT'] = answer

    #########################################################################################

    if properties['CALCULATOR'] != 'XTB':
        answer = ask((f'Current command to call {properties["CALCULATOR"]} is {COMMANDS[properties["CALCULATOR"]]}. ' +
                    'If you would like to change it, type it here, otherwise press enter : '), accepted=None)
        if answer != '':
            properties['NEW_COMMAND'] = answer

    #########################################################################################

    if properties['CALCULATOR'] in ('ORCA', 'GAUSSIAN'):
        properties['PROCS'] = ask(f'How many cores should {properties["CALCULATOR"]} jobs run on? [1] : ',
                                    accepted=[str(n) for n in range(1,1000)], default=1)

    #########################################################################################

    if properties['CALCULATOR'] == 'GAUSSIAN':
        properties['MEM_GB'] = int(ask(f'How much memory should a GAUSSIAN job have, in GBs? [1] : ',
                                    accepted=[str(n) for n in range(1,1000)], default='1'))

    #########################################################################################

    rank = {
        'MOPAC':1,
        'ORCA':2,
        'GAUSSIAN':3,
        'XTB':4,
    }

    q = "\'"

    with open('settings.py', 'r') as f:
        lines = f.readlines()

    old_lines = lines.copy()

    for l, line in enumerate(old_lines):

        if 'OPENBABEL_OPT_BOOL =' in line:
            lines[l] = 'OPENBABEL_OPT_BOOL = ' + str(properties['OPENBABEL_OPT_BOOL']) + '\n'

        elif 'CALCULATOR =' in line:
            lines[l] = 'CALCULATOR = ' + q + properties['CALCULATOR'] + q + '\n'

        elif 'DEFAULT_LEVELS = {' in line:
            if properties['NEW_DEFAULT'] is not None:
                lines[l+rank[properties['CALCULATOR']]] = ' '*4 + q + properties['CALCULATOR'] + q + ':' + q + properties['NEW_DEFAULT'] + q + ',\n'

        elif 'COMMANDS = {' in line:
            if properties['NEW_COMMAND'] is not None:
                lines[l+rank[properties['CALCULATOR']]] = ' '*4 + q + properties['CALCULATOR'] + q + ':' + q + properties['NEW_COMMAND'] + q + ',\n'

        elif 'PROCS =' in line:
            lines[l] = 'PROCS = ' + str(properties['PROCS']) + '\n'

        elif 'MEM_GB =' in line:
            lines[l] = 'MEM_GB = ' + str(properties['MEM_GB']) + '\n'

    with open('settings.py', 'w') as f:
        f.write(''.join(lines))




