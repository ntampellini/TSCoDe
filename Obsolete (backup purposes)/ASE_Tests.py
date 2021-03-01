from ase import Atoms
from ase.visualize import view
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from ase.calculators.mopac import MOPAC
from cclib.io import ccread
from reactive_atoms_classes import pt
import os, time

class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in 
    Python, i.e. will suppress all print, even if the print originates in a 
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).      

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

os.chdir(os.path.dirname(os.path.realpath(__file__)))

os.chdir('mopac_tests')

# data = ccread('TS_out.xyz')
data = ccread('optimized.xyz')

methods = ['AM1', 'MNDO', 'MNDOD', 'PM3', 'PM6', 'PM6-D3', 'PM6-DH+',
           'PM6-DH2', 'PM6-DH2X', 'PM6-D3H4', 'PM6-D3H4X', 'PM7','RM1']

for method in methods:

    atoms = Atoms(''.join([pt[i].symbol for i in data.atomnos]), positions=data.atomcoords[0])

    # atoms.edit()
    atoms.set_constraint(FixAtoms(indices=[6,18]))

    jobname = 'TEST_TS'
    atoms.calc = MOPAC(label=jobname, command=f'mopac2016 {jobname}.mop', method=method)
    opt = BFGS(atoms, trajectory=f'{jobname}.traj', logfile=f'{jobname}.log')

    t_start = time.time()

    with suppress_stdout_stderr():
        opt.run(fmax=0.05)

    t_end = time.time()

    print(method, round(t_end-t_start, 3), 's')

view(atoms)