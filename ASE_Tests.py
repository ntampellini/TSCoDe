from ase import Atoms
from ase.visualize import view
from ase.optimize import BFGS
from ase.calculators.mopac import MOPAC

atoms = Atoms('NN', positions=[[0, 0, -1], [0, 0, 1]])


constraints = [FixAtoms(indices=[0, 1, 2]),
               Hookean(a1=8, a2=9, rt=2.6, k=15.),
               Hookean(a1=8, a2=(0., 0., 1., -15.), k=15.), ]
atoms.set_constraint(constraints)


atoms.calc = MOPAC(label='N2', command='PM7-TS')
print(atoms.get_potential_energy())


# opt = BFGS(atoms, trajectory='opt.traj', logfile='opt.log')
# opt.run(fmax=0.05)

# view('opt.traj')