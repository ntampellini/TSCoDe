# %%
import os
from cclib.io import ccread
os.chdir(r'C:\Users\Nik\Desktop\Coding\TSCoDe\Resources\DA\ase')
data = ccread('structure_0_start.xyz')
start = data.atomcoords[0]
end = ccread('structure_0_end.xyz').atomcoords[0]


# %%
import time

class timer():
    def __init__(self, title=''):
        self.title = title
    def __enter__(self):
        self.t1 = time.time()
    def __exit__(self, type, value, traceback):
        self.t2 = time.time()
        if self.title != '':
            s = f'{self.title} - {round(self.t2-self.t1, 3)} s'
        else:
            s = f'{round(self.t2-self.t1, 3)} s' 
        print(s)


# %%
from ase.dyneb import DyNEB
from ase.neb import NEB
from ase.optimize import *
from ase import Atoms
from ase.calculators.mopac import MOPAC
from parameters import MOPAC_COMMAND
from ase.visualize import view
from optimization_methods import write_xyz

def dump(filename, images, atomnos):
    with open(filename, 'w') as f:
                for image in images:
                    coords = image.get_positions()
                    write_xyz(coords, atomnos, f)

def ase_neb(reagents, products, atomnos, n_images=6, fmax=0.05, label='temp', optimizer='LFBGS'):

    # Read initial and final states:
    initial = Atoms(atomnos, positions=reagents)
    final = Atoms(atomnos, positions=products)

    # Make a band consisting of n images:
    images = [initial]
    images += [initial.copy() for i in range(n_images)]
    images += [final]

    climbing_neb = DyNEB(images, fmax=fmax, climb=True,  method='eb', scale_fmax=1, allow_shared_calculator=True)

    # Interpolate linearly the positions of the middle images:
    climbing_neb.interpolate()
    # dump('interpolated.xyz', images, atomnos)
    
    # Set calculators for all images
    for i, image in enumerate(images):
        image.calc = MOPAC(label=label, command=f'{MOPAC_COMMAND} {label}.mop', method='PM7 GEO-OK')

    # Set the optimizer
    if optimizer == 'BFGS':
        climbing_optimizer = BFGS(climbing_neb)
    elif optimizer == 'LFBGS':
        climbing_optimizer = LBFGS(climbing_neb)
    elif optimizer == 'FIRE':
        climbing_optimizer = FIRE(climbing_neb)
    elif optimizer == 'MDMin':
        climbing_optimizer = MDMin(climbing_neb)

    # Optimize:
    try:
        climbing_optimizer.run(fmax=fmax, steps=500)

    except Exception as e:
        print(e)

    # dump('debug.xyz', images, atomnos)
    energies = [image.get_total_energy() for image in images]
    ts_id = energies.index(max(energies))
    # print(f'TS structure is number {ts_id}, energy is {max(energies)}')

    return images[ts_id].get_positions()


# %%
# ts_coords = ase_neb(start, end, data.atomnos)


# %%
# from optimization_methods import mopac_opt

# mopac_opt(ts_coords, data.atomnos, method='PM7 FORCE', title='ts', read_output=False)
# os.system('avogadro ts.out')
# test the obtained TS (force MOPAC calc)


# %%
from tscode import suppress_stdout_stderr

t = []
opts = ('LFBGS', 'BFGS', 'MDMin')
for opt in opts:
    t1 = time.time()
    ase_neb(start, end, data.atomnos, optimizer=opt)
    t2 = time.time()
    t.append(round(t2-t1, 3))

# %%
for i in range(len(t)):
    print(f'{opts[i]} - {t[i]}')



# %%
