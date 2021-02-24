'''
TSCoDe - Transition state Seeker from Conformational Density

(Work in Progress)

'''
from hypermolecule_class import Hypermolecule, pt
import numpy as np
from copy import deepcopy
from parameters import *
from pprint import pprint
from scipy.spatial.transform import Rotation as R
import os
import time
from cclib.io import ccread
from openbabel import openbabel as ob
from ase import Atoms
from ase.visualize import view
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from ase.calculators.mopac import MOPAC


from subprocess import DEVNULL, STDOUT, check_call


def loadbar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='#'):
	percent = ('{0:.' + str(decimals) + 'f}').format(100 * (iteration/float(total)))
	filledLength = int(length * iteration // total)
	bar = fill * filledLength + '-' * (length - filledLength)
	print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
	if iteration == total:
		print()

def norm(vec):
    return vec / np.linalg.norm(vec)

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
        second_atom = mol.GetAtom(constrained_indexes[1]+1)
        length = first_atom.GetDistance(second_atom)

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

    return atoms.positions, atoms.get_total_energy

def cartesian_product(*arrays):
    return np.stack(np.meshgrid(*arrays), -1).reshape(-1, len(arrays))

def calc_positioned_conformers(self):
    self.positioned_conformers = np.array([[self.rotation @ v + self.position for v in conformer] for conformer in self.atomcoords])

def write_xyz(coords:np.array, atomnos:np.array, output, title='TEST'):
    '''
    output is of _io.TextIOWrapper type

    '''
    assert atomnos.shape[0] == coords.shape[0]
    assert coords.shape[1] == 3
    string = ''
    string += str(len(coords))
    string += f'\n{title}\n'
    for i, atom in enumerate(coords):
        string += '%s\t%s %s %s\n' % (pt[atomnos[i]].symbol, round(atom[0], 6), round(atom[1], 6), round(atom[2], 6))
    output.write(string)

def rotation_matrix_from_vectors(vec1, vec2):
    """
    Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.

    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

class Docker:
    def __init__(self, *objects):
        self.objects = list(*objects)
        self.objects = sorted(self.objects, key=lambda x: len(x.atomnos), reverse=True)
    
    def setup(self, repeat=1, maxcycles=100):
        self.repeat = repeat
        self.maxcycles = maxcycles

    def run(self, debug=False):
        '''
        '''
        print()       

        t_start = time.time()

        cyclical = False
        if all([len(molecule.reactive_atoms_classes) == 2 for molecule in self.objects]) or len(self.objects) > 2:
            cyclical = True

        centers_indexes = cartesian_product(*[np.array(range(len(molecule.centers))) for molecule in self.objects])
        # for two mols with 3 and 2 centers: [[0 0][0 1][1 0][1 1][2 0][2 1]]
        
        threads = []
        for _ in range(len(centers_indexes)*self.repeat):
            threads.append([deepcopy(obj) for obj in self.objects])

        for t, thread in enumerate(threads): # each run is a different "regioisomer", repeated self.repeat times

            indexes = centers_indexes[t % len(centers_indexes)] # looping over the indexes that define "regiochemistry"
            repeated = True if t // len(centers_indexes) > 0 else False

            if cyclical:
                raise Exception('Cyclical TS: still to be implemented. Sorry.')

            else:
                for i, molecule in enumerate(thread[1:]): #first molecule is always frozen in place, other(s) are placed with an orbital criterion

                    ref_orb_vec = thread[i].centers[indexes[i]]  # absolute, arbitrarily long
                    # reference molecule is the one just before
                    mol_orb_vec = molecule.centers[indexes[i+1]]

                    ref_orb_vers = thread[i].orb_vers[indexes[i]]  # unit length, relative to orbital orientation
                    # reference molecule is the one just before
                    mol_orb_vers = molecule.orb_vers[indexes[i+1]]

                    molecule.rotation = rotation_matrix_from_vectors(mol_orb_vers, -ref_orb_vers)

                    molecule.position = thread[i].rotation @ ref_orb_vec + thread[i].position - molecule.rotation @ mol_orb_vec

                    if repeated:

                        pointer = molecule.rotation @ mol_orb_vers

                        rotation = (np.random.rand()*2. - 1.) * np.pi    # random angle (radians) between -pi and pi
                        quat = np.array([np.sin(rotation/2)*pointer[0],
                                         np.sin(rotation/2)*pointer[1],
                                         np.sin(rotation/2)*pointer[2],
                                         np.cos(rotation/2)])            # normalized quaternion, scalar last (i j k w)

                        delta_rot = R.from_quat(quat).as_matrix()
                        molecule.rotation = delta_rot @ molecule.rotation

                        molecule.position = thread[i].rotation @ ref_orb_vec + thread[i].position - molecule.rotation @ mol_orb_vec

                        # print(f'Random rotation of {rotation / np.pi * 180} degrees performed on candidate {t}')

############################################################################################################################

            # with open('ouroboros_setup.xyz', 'a') as f:
            #     structure = np.array([thread[0].rotation @ v + thread[0].position for v in thread[0].hypermolecule])
            #     atomnos = thread[0].hypermolecule_atomnos
            #     for molecule in thread[1:]:
            #         s = np.array([molecule.rotation @ v + molecule.position for v in molecule.hypermolecule])
            #         structure = np.concatenate((structure, s))
            #         atomnos = np.concatenate((atomnos, molecule.hypermolecule_atomnos))
            #     write_xyz(structure, atomnos, f, title=f'Arrangement_{t}')
        # quit()

############################################################################################################################

        atomnos = np.concatenate([molecule.atomnos for molecule in objects])
        # just a way not to lose track of atomic numbers associated with coordinates

        try:
            os.remove('TS_out.xyz')
        except:
            pass
        
        # GENERATING ALL POSSIBLE COMBINATIONS OF CONFORMATIONS AND STORING THEM IN SELF.STRUCTURES

        conf_number = [len(molecule.atomcoords) for molecule in objects]
        conf_indexes = cartesian_product(*[np.array(range(i)) for i in conf_number])
        # first index of each vector is the conformer number of the first molecule and so on...

        self.structures = np.zeros((int(len(conf_indexes)*int(len(threads))), len(atomnos), 3)) # like atomcoords property, but containing multimolecular arrangements

        for geometry_number, geometry in enumerate(threads):

            for molecule in geometry:
                calc_positioned_conformers(molecule)

            for i, conf_index in enumerate(conf_indexes): # 0, [0,0,0] then 1, [0,0,1] then 2, [0,1,1]
                count_atoms = 0

                for molecule_number, conformation in enumerate(conf_index): # 0, 0 then 1, 0 then 2, 0 (first [] of outer for loop)
                    coords = geometry[molecule_number].positioned_conformers[conformation]
                    n = len(geometry[molecule_number].atomnos)
                    self.structures[geometry_number*len(conf_indexes)+i][count_atoms:count_atoms+n] = coords
                    count_atoms += n

        print(f'Generated {len(self.structures)} transition state candidates')

        ################################################# START OF STRUCTURAL OPTIMIZATION

        constrained_indexes = [int(self.objects[0].reactive_indexes[0]), int(self.objects[1].reactive_indexes[0] + len(self.objects[0].atomcoords[0]))]
        
        if debug: print('Constrained indexes are', constrained_indexes)

        for i, structure in enumerate(deepcopy(self.structures)):
            loadbar(i, len(self.structures), prefix=f'Optimizing structure {i}/{len(self.structures)} ')
            self.structures[i], self.energies = MM_SE_POPT(structure, atomnos, constrained_indexes)
        loadbar(1, 1, prefix=f'Optimizing structure {len(self.structures)}/{len(self.structures)} ')




        with open('TS_out.xyz', 'w') as f:        
            for i, structure in enumerate(self.structures):
                write_xyz(structure, atomnos, f, title=f'TS candidate {i}')




try:
    os.remove('ouroboros_setup.xyz')
except:
    pass

a = ['Resources/SN2/MeOH_ensemble.xyz', 1]
# a = ['Resources/dienamine/dienamine_ensemble.xyz', 7]
# a = ['Resources/SN2/amine_ensemble.xyz', 22]
# a = ['Resources/indole/indole_ensemble.xyz', 7]


b = ['Resources/SN2/CH3Br_ensemble.xyz', 0]
# b = ['Resources/SN2/ketone_ensemble.xyz', 3]


inp = [a,b]

objects = [Hypermolecule(m[0], m[1]) for m in inp]

docker = Docker(objects) # initialize docker with molecule density objects
docker.setup(repeat=2, maxcycles=50) # set variables

os.chdir('Resources/SN2')

docker.run(debug=True)

path = os.path.join(os.getcwd(), 'TS_out.vmd')
check_call(f'vmd -e {path}'.split(), stdout=DEVNULL, stderr=STDOUT)