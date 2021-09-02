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
epsilon_dict = {
    'aceticacid':6.15,
    'ch3cooh':6.15,
    'acetone':20.7,
    'acetonitrile':37.5,
    'ch3cn':37.5,
    'aniline':7.06,
    'benzaldehyde':17.9,
    'benzene':2.28,
    'chloroform':4.8,
    'ch3cl':4.8,
    'carbondisulphide':2.63,
    'cs2':2.63,
    'dcm':8.93,
    'ch2cl2':8.93,
    'dioxane':2.25,
    'dmf':36.71,
    'dmso':46.68,
    'diethylether':4.27,
    'et2o':4.27,
    'dimethylether':6.18,
    'ethanol':24.3,
    'etoh':24.3,
    'methanol':32.63,
    'ch3oh':32.63,
    'meoh':32.63,
    'ethylacetate':6.02,
    'furan':2.94,
    'hexadecane':2.05,
    'octanol':10.30,
    'phenol':12.4,
    'toluene':2.38,
    'thf':7.58,
    'water':80.1,
    'h2o':80.1,
}

new_theory_level = {
    'MOPAC':lambda theory_level, solvent: f'EPS={epsilon_dict[solvent]}',
    'GAUSSIAN':lambda theory_level, solvent: f'scrf=(cpcm,solvent={solvent})',
    'ORCA':lambda theory_level, solvent: f'! CPCM\n%cpcm\nepsilon {epsilon_dict[solvent]}\nend',
    # 'XTB':lambda theory_level, _: '',
}

def get_solvent_line(solvent, calculator, theory_level):

    if solvent is None:
        return ''

    if solvent not in epsilon_dict:
        print(f'Solvent \'{solvent}\' not recognized. Implemented solvents are:')
        for solvent in epsilon_dict:
            print('    '+solvent)
        print('Please note that not all solvents will work with all calculators.')
        quit()

    return new_theory_level[calculator](theory_level, solvent)

if __name__ == '__main__':
    print(get_solvent_line('aceticacid', 'ORCA', 'PM7'))