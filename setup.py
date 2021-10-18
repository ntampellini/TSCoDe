from setuptools import setup, find_packages

long_description = ('## TSCoDe: Transition State Conformational Docker.\nSystematically generate poses for ' +
                'bimolecular and trimolecular transition states. Support for open and cyclical transition ' +
                'states, exploring all regiosomeric and stereoisomeric poses.')

with open('CHANGELOG.md', 'r') as f:
    long_description += '\n\n'
    long_description += f.read()

setup(
    name='tscode',
    version='0.0.5',
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
        'numpy',
        'scipy',
        'cclib',
        'periodictable',
        'matplotlib',
        'networkx',
        'rmsd',
        'ase',
        'sella',
        'sklearn',
        'numba',
    ],

    url='https://www.github.com/ntampellini/tscode',
    author='Nicolò Tampellini',
    author_email='nicolo.tampellini@yale.edu',

    packages=find_packages(),
    python_requires=">=3.8",
)