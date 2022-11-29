# coding=utf-8
'''

TSCODE: Transition State Conformational Docker
Copyright (C) 2021-2022 Nicolò Tampellini

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

'''
from tscode.__main__ import __version__
from setuptools import setup, find_packages

long_description = ('## TSCoDe: Transition State Conformational Docker.\nSystematically generate poses for ' +
                'bimolecular and trimolecular transition states. Support for open and cyclical transition ' +
                'states, exploring all regiosomeric and stereoisomeric poses.')

with open('CHANGELOG.md', 'r') as f:
    long_description += '\n\n'
    long_description += f.read()

setup(
    name='tscode',
    version=__version__,
    description='Computational chemistry general purpose transition state builder',
    keywords=['computational chemistry', 'ASE', 'transition state', 'xtb'],

    # package_dir={'':'tscode'},
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],

    long_description=long_description,
    long_description_content_type='text/markdown',

    install_requires=[
        'numpy==1.20.3',
        'scipy==1.6.2',
        'numba-scipy==0.3.1',
        'cclib==1.7',
        'periodictable==1.6.0',
        'matplotlib==3.4.3',
        'networkx==2.5.1',
        'rmsd==1.4',
        'ase==3.21.1',
        'sella',
        'sklearn',
        'numba==0.54.0',
        'prettytable==3.3.0'
    ],

    url='https://www.github.com/ntampellini/tscode',
    author='Nicolò Tampellini',
    author_email='nicolo.tampellini@yale.edu',

    packages=find_packages(),
    python_requires=">=3.7",
)