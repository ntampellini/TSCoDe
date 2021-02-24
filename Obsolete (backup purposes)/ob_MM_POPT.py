# Adapted by https://gist.github.com/andersx/7784817

from openbabel import openbabel as ob
import os

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


def MM_SE_POPT(coords, atomnos, constrained_indexes, methods=['UFF','PM7'], debug=False):
    '''
    Performs a two-step constrained optimizaiton, first MM with OpenBabel,
    then a semiempirical calculation with MOPAC. Methods can be:

    [['UFF','MMFF'], ['AM1', 'MNDO', 'MNDOD', 'PM3', 'PM6', 'PM6-D3', 'PM6-DH+', 
                      'PM6-DH2', 'PM6-DH2X', 'PM6-D3H4', 'PM6-D3H4X', 'PM7','RM1']]

    return : optimized coordinates

    '''

    def OB_MM_POPT(filename, constrained_indexes, method):
        '''
        return : name of MM-optimized outfile

        '''

        assert len(constrained_indexes) == 2

        rootname, extension = filename.split('.')

        # Standard openbabel molecule load
        conv = ob.OBConversion()
        conv.SetInAndOutFormats(extension,'xyz')
        mol = ob.OBMol()
        conv.ReadFile(mol, filename)

        # Define constraints

        first_atom = mol.GetAtom(constrained_indexes[0]+1)
        length = first_atom.GetDistance(constrained_indexes[1]+1)

        constraints = ob.OBFFConstraints()
        constraints.AddDistanceConstraint(constrained_indexes[0]+1, constrained_indexes[1]+1, length)       # Angstroms
        # constraints.AddAngleConstraint(1, 2, 3, 120.0)      # Degrees
        # constraints.AddTorsionConstraint(1, 2, 3, 4, 180.0) # Degrees

        # Setup the force field with the constraints
        forcefield = ob.OBForceField.FindForceField(method)
        forcefield.Setup(mol, constraints)
        forcefield.SetConstraints(constraints)

        # Do a 500 steps conjugate gradient minimiazation
        # and save the coordinates to mol.
        forcefield.ConjugateGradients(500)
        forcefield.GetCoordinates(mol)

        # Write the mol to a file
        outname = rootname + '_MM.xyz'
        conv.WriteFile(mol,outname)

        return outname

    assert len(constrained_indexes) == 2

    MM_name = 'temp.xyz'

    with open(MM_name, 'w') as f:
        write_xyz(coords, atomnos, f, 'Intermediate file to be fed to OBabel')

    data = ccread(OB_MM_POPT(MM_name, constrained_indexes, methods[0]))
    os.remove(MM_name)

    # methods = ['AM1', 'MNDO', 'MNDOD', 'PM3', 'PM6', 'PM6-D3', 'PM6-DH+',
    #         'PM6-DH2', 'PM6-DH2X', 'PM6-D3H4', 'PM6-D3H4X', 'PM7','RM1']


    atoms = Atoms(''.join([pt[i].symbol for i in data.atomnos]), positions=data.atomcoords[0])

    atoms.set_constraint(FixAtoms(indices=[constrained_indexes[0],constrained_indexes[1]]))

    jobname = 'TEST_TS'
    atoms.calc = MOPAC(label=jobname, command=f'mopac2016 {jobname}.mop', method=methods[1])
    opt = BFGS(atoms, trajectory=f'{jobname}.traj', logfile=f'{jobname}.log')

    t_start = time.time()

    with suppress_stdout_stderr():
        opt.run(fmax=0.05)

    t_end = time.time()

    if debug: print(f'Optimization at {method} level took', round(t_end-t_start, 3), 's')

    return atoms.positions











if __name__ == '__main__':
    OB_MM_POPT('TS_out.xyz', [6,18])