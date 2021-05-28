from copy import deepcopy
import os
import numpy as np
from ase import Atoms
from ase.calculators.mopac import MOPAC
from ase.dyneb import DyNEB
from ase.optimize import BFGS, LBFGS
from hypermolecule_class import pt, graphize
from parameters import MOPAC_COMMAND
from utils import *
from subprocess import DEVNULL, STDOUT, check_call
from parameters import nci_dict
import itertools as it
from rmsd import kabsch
from openbabel import openbabel as ob
from cclib.io import ccread

class MopacReadError(Exception):
    '''
    Thrown when reading MOPAC output files fails for some reason.
    '''
    pass

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

class Spring:
    '''
    ASE Custom Constraint Class
    Adds an harmonic force between a pair of atoms.
    Spring constant is very high to achieve tight convergence,
    but force is dampened so as not to ruin structure.
    '''
    def __init__(self, i1, i2, d_eq, k=1000):
        self.i1, self.i2 = i1, i2
        self.d_eq = d_eq
        self.k = k

    def adjust_positions(self, atoms, newpositions):
        pass

    def adjust_forces(self, atoms, forces):

        direction = atoms.positions[self.i2] - atoms.positions[self.i1]
        # vector connecting atom1 to atom2

        spring_force = self.k * (np.linalg.norm(direction) - self.d_eq)
        # absolute spring force (float). Positive if spring is overstretched.

        spring_force = np.clip(spring_force, -100, 100)
        # force is clipped at 100 eV/A

        forces[self.i1] += (norm(direction) * spring_force)
        forces[self.i2] -= (norm(direction) * spring_force)
        # applying harmonic force to each atom, directed toward the other one

    def __repr__(self):
        return f'Spring - ids:{self.i1}/{self.i2} - d_eq:{self.d_eq}, k:{self.k}'

def write_xyz(coords:np.array, atomnos:np.array, output, title='temp'):
    '''
    output is of _io.TextIOWrapper type

    '''
    assert atomnos.shape[0] == coords.shape[0]
    assert coords.shape[1] == 3
    string = ''
    string += str(len(coords))
    string += f'\n{title}\n'
    for i, atom in enumerate(coords):
        string += '%s     % .6f % .6f % .6f\n' % (pt[atomnos[i]].symbol, atom[0], atom[1], atom[2])
    output.write(string)

def openbabel_opt(structure, atomnos, constrained_indexes, method='UFF'):
        '''
        return : MM-optimized structure (UFF/MMFF)
        '''

        filename='temp_ob_in.xyz'

        with open(filename, 'w') as f:
            write_xyz(structure, atomnos, f)

        outname = 'temp_ob_out.xyz'

        # Standard openbabel molecule load
        conv = ob.OBConversion()
        conv.SetInAndOutFormats('xyz','xyz')
        mol = ob.OBMol()
        more = conv.ReadFile(mol, filename)
        i = 0

        # Define constraints
        constraints = ob.OBFFConstraints()

        for a, b in constrained_indexes:

            first_atom = mol.GetAtom(int(a+1))
            length = first_atom.GetDistance(int(b+1))

            constraints.AddDistanceConstraint(int(a+1), int(b+1), length)       # Angstroms
            # constraints.AddAngleConstraint(1, 2, 3, 120.0)      # Degrees
            # constraints.AddTorsionConstraint(1, 2, 3, 4, 180.0) # Degrees

        # Setup the force field with the constraints
        forcefield = ob.OBForceField.FindForceField(method)
        forcefield.Setup(mol, constraints)
        forcefield.SetConstraints(constraints)

        # Do a 500 steps conjugate gradient minimization
        # (or less if converges) and save the coordinates to mol.
        forcefield.ConjugateGradients(500)
        forcefield.GetCoordinates(mol)

        # Write the mol to a file
        conv.WriteFile(mol,outname)
        conv.CloseOutFile()

        opt_coords = ccread(outname).atomcoords[0]

        return opt_coords

def scramble(array, sequence):
    return np.array([array[s] for s in sequence])

def read_mop_out(filename):
    '''
    Reads a MOPAC output looking for optimized coordinates and energy.
    :params filename: name of MOPAC filename (.out extension)
    :return coords, energy: array of optimized coordinates and absolute energy, in kcal/mol
    '''
    coords = []
    with open(filename, 'r') as f:
        while True:
            line = f.readline()

            if 'Too many variables. By definition, at least one force constant is exactly zero' in line:
                success = False
                return None, np.inf, success

            if not line:
                break
            
            if 'SCF FIELD WAS ACHIEVED' in line:
                    while True:
                        line = f.readline()
                        if not line:
                            break

                        if 'FINAL HEAT OF FORMATION' in line:
                            energy = float(line.split()[5])
                            # in kcal/mol

                        if 'CARTESIAN COORDINATES' in line:
                            line = f.readline()
                            line = f.readline()
                            while line != '\n':
                                splitted = line.split()
                                # symbols.append(splitted[1])
                                coords.append([float(splitted[2]),
                                               float(splitted[3]),
                                               float(splitted[4])])
                                            
                                line = f.readline()
                                if not line:
                                    break
                            break
                    break
    coords = np.array(coords)
    if coords.shape[0] != 0:
        success = True
        return coords, energy, success
    else:
        raise MopacReadError(f'Cannot read file {filename}: maybe a badly specified MOPAC keyword?')

def mopac_opt(coords, atomnos, constrained_indexes=None, method='PM7', title='temp', read_output=True):
    '''
    This function writes a MOPAC .mop input, runs it with the subprocess
    module and reads its output. Coordinates used are mixed
    (cartesian and internal) to be able to constrain the reactive atoms
    distances specified in constrained_indexes.

    :params coords: array of shape (n,3) with cartesian coordinates for atoms
    :params atomnos: array of atomic numbers for atoms
    :params constrained_indexes: array of shape (n,2), with the indexes
                                 of atomic pairs to be constrained
    :params method: string, specifiyng the first line of keywords for the MOPAC input file.
    :params title: string, used as a file name and job title for the mopac input file.
    '''

    constrained_indexes_list = constrained_indexes.ravel() if constrained_indexes is not None else []
    constrained_indexes = constrained_indexes if constrained_indexes is not None else []

    order = []
    s = [method + '\n' + title + '\n\n']
    for i, num in enumerate(atomnos):
        if i not in constrained_indexes:
            order.append(i)
            s.append(' {} {} 1 {} 1 {} 1\n'.format(pt[num].symbol, coords[i][0], coords[i][1], coords[i][2]))

    free_indexes = list(set(range(len(atomnos))) - set(constrained_indexes_list))
    # print('free indexes are', free_indexes, '\n')

    if len(constrained_indexes_list) == len(set(constrained_indexes_list)):
    # block pairs of atoms if no atom is involved in more than one distance constrain

        for a, b in constrained_indexes:
            
            order.append(b)
            order.append(a)

            c, d = np.random.choice(free_indexes, 2)
            while c == d:
                c, d = np.random.choice(free_indexes, 2)
            # indexes of reference atoms, from unconstraind atoms set

            dist = np.linalg.norm(coords[a] - coords[b]) # in Angstrom
            # print(f'DIST - {dist} - between {a} {b}')

            angle = np.arccos(norm(coords[a] - coords[b]) @ norm(coords[c] - coords[b]))*180/np.pi # in degrees
            # print(f'ANGLE - {angle} - between {a} {b} {c}')

            d_angle = dihedral([coords[a],
                                coords[b],
                                coords[c],
                                coords[d]])
            d_angle += 360 if d_angle < 0 else 0
            # print(f'D_ANGLE - {d_angle} - between {a} {b} {c} {d}')

            list_len = len(s)
            s.append(' {} {} 1 {} 1 {} 1\n'.format(pt[atomnos[b]].symbol, coords[b][0], coords[b][1], coords[b][2]))
            s.append(' {} {} 0 {} 1 {} 1 {} {} {}\n'.format(pt[atomnos[a]].symbol, dist, angle, d_angle, list_len, free_indexes.index(c)+1, free_indexes.index(d)+1))
            # print(f'Blocked bond between mopac ids {list_len} {list_len+1}\n')

    elif len(set(constrained_indexes_list)) == 3:
    # three atoms, the central bound to the other two
    # OTHERS[0]: cartesian
    # CENTRAL: internal (self, others[0], two random)
    # OTHERS[1]: internal (self, central, two random)
        
        central = max(set(constrained_indexes_list), key=lambda x: list(constrained_indexes_list).count(x))
        # index of the atom that is constrained to two other

        others = list(set(constrained_indexes_list) - {central})

    # OTHERS[0]

        order.append(others[0])
        s.append(' {} {} 1 {} 1 {} 1\n'.format(pt[atomnos[others[0]]].symbol, coords[others[0]][0], coords[others[0]][1], coords[others[0]][2]))
        # first atom is placed in cartesian coordinates, the other two have a distance constraint and are expressed in internal coordinates

    #CENTRAL

        order.append(central)
        c, d = np.random.choice(free_indexes, 2)
        while c == d:
            c, d = np.random.choice(free_indexes, 2)
        # indexes of reference atoms, from unconstraind atoms set

        dist = np.linalg.norm(coords[central] - coords[others[0]]) # in Angstrom

        angle = np.arccos(norm(coords[central] - coords[others[0]]) @ norm(coords[others[0]] - coords[c]))*180/np.pi # in degrees

        d_angle = dihedral([coords[central],
                            coords[others[0]],
                            coords[c],
                            coords[d]])
        d_angle += 360 if d_angle < 0 else 0

        list_len = len(s)
        s.append(' {} {} 0 {} 1 {} 1 {} {} {}\n'.format(pt[atomnos[central]].symbol, dist, angle, d_angle, list_len-1, free_indexes.index(c)+1, free_indexes.index(d)+1))

    #OTHERS[1]

        order.append(others[1])
        c1, d1 = np.random.choice(free_indexes, 2)
        while c1 == d1:
            c1, d1 = np.random.choice(free_indexes, 2)
        # indexes of reference atoms, from unconstraind atoms set

        dist1 = np.linalg.norm(coords[others[1]] - coords[central]) # in Angstrom

        angle1 = np.arccos(norm(coords[others[1]] - coords[central]) @ norm(coords[others[1]] - coords[c1]))*180/np.pi # in degrees

        d_angle1 = dihedral([coords[others[1]],
                             coords[central],
                             coords[c1],
                             coords[d1]])
        d_angle1 += 360 if d_angle < 0 else 0

        list_len = len(s)
        s.append(' {} {} 0 {} 1 {} 1 {} {} {}\n'.format(pt[atomnos[others[1]]].symbol, dist1, angle1, d_angle1, list_len-1, free_indexes.index(c1)+1, free_indexes.index(d1)+1))

    else:
        raise NotImplementedError('The constraints provided for MOPAC optimization are not yet supported')


    s = ''.join(s)
    with open(f'{title}.mop', 'w') as f:
        f.write(s)
    
    try:
        check_call(f'{MOPAC_COMMAND} {title}.mop'.split(), stdout=DEVNULL, stderr=STDOUT)
    except KeyboardInterrupt:
        print('KeyboardInterrupt requested by user. Quitting.')
        quit()

    os.remove(f'{title}.mop')
    # delete input, we do not need it anymore

    if read_output:

        inv_order = [order.index(i) for i in range(len(order))]
        # undoing the atomic scramble that was needed by the mopac input requirements

        opt_coords, energy, success = read_mop_out(f'{title}.out')
        os.remove(f'{title}.out')

        opt_coords = scramble(opt_coords, inv_order) if opt_coords is not None else coords
        # If opt_coords is None, that is if TS seeking crashed,
        # sets opt_coords to the old coords. If not, unscrambles
        # coordinates read from mopac output.

        return opt_coords, energy, success

def optimize(TS_structure, TS_atomnos, mols_graphs, constrained_indexes=None, method='PM7 GEO-OK', max_newbonds=2, title='temp', debug=False):
    '''
    Performs a geometry partial optimization (POPT) with Mopac at $method level, 
    constraining the distance between the specified atom pair. Moreover, performs a check of atomic
    pairs distances to ensure to have preserved molecular identities and prevented atom scrambling.

    :params TS_structure: list of coordinates for each atom in the TS
    :params TS_atomnos: list of atomic numbers for each atom in the TS
    :params constrained_indexes: indexes of constrained atoms in the TS geometry
    :params mols_graphs: list of molecule.graph objects, containing connectivity information
    :params method: Level of theory to be used in geometry optimization. Default if UFF.

    :return opt_coords: optimized structure
    :return energy: absolute energy of structure, in kcal/mol
    :return not_scrambled: bool, indicating if the optimization shifted up some bonds (except the constrained ones)
    '''
    assert len(TS_structure) == sum([len(graph.nodes) for graph in mols_graphs])

    if constrained_indexes is None:
        constrained_indexes = np.array(())

    opt_coords, energy, success = mopac_opt(TS_structure, TS_atomnos, constrained_indexes, method=method, title=title)

    if success:
        success = scramble_check(opt_coords, TS_atomnos, mols_graphs, max_newbonds=max_newbonds)

    return opt_coords, energy, success

def molecule_check(old_coords, new_coords, atomnos, max_newbonds=1):
    '''
    Checks if two molecules have the same bonds between the same atomic indexes
    '''
    old_bonds = {(a, b) for a, b in list(graphize(old_coords, atomnos).edges) if a != b}
    new_bonds = {(a, b) for a, b in list(graphize(new_coords, atomnos).edges) if a != b}

    delta_bonds = (old_bonds | new_bonds) - (old_bonds & new_bonds)

    if len(delta_bonds) > max_newbonds:
        return False
    else:
        return True

def scramble_check(TS_structure, TS_atomnos, mols_graphs, max_newbonds=1) -> bool:
    '''
    Check if a transition state structure has scrambled during some optimization
    steps. If more than a given number of bonds changed (formed or broke) the
    structure is considered scrambled, and the method returns False.
    '''
    assert len(TS_structure) == sum([len(graph.nodes) for graph in mols_graphs])

    bonds = []
    for i, graph in enumerate(mols_graphs):

        pos = 0
        while i != 0:
            pos += len(mols_graphs[i-1].nodes)
            i -= 1

        for bond in [(a+pos, b+pos) for a, b in list(graph.edges) if a != b]:
            bonds.append(bond)

    bonds = set(bonds)
    # creating bond set containing all bonds present in the desired transition state

    new_bonds = {(a, b) for a, b in list(graphize(TS_structure, TS_atomnos).edges) if a != b}
    delta_bonds = (bonds | new_bonds) - (bonds & new_bonds)

    if len(delta_bonds) > max_newbonds:
        return False
    else:
        return True

def dump(filename, images, atomnos):
    with open(filename, 'w') as f:
                for i, image in enumerate(images):
                    coords = image.get_positions()
                    write_xyz(coords, atomnos, f, title=f'{filename[:-4]}_image_{i}')

def ase_adjust_spacings(self, structure, atomnos, mols_graphs, method='PM7', max_newbonds=2):
    '''
    TODO - desc
    '''
    atoms = Atoms(atomnos, positions=structure)
    atoms.calc = MOPAC(label='temp', command=f'{MOPAC_COMMAND} temp.mop > temp.cmdlog 2>&1', method=method)
    
    springs = []
    for i, dist in enumerate([dist for _, dist in self.pairings_dists]):
        i1, i2 = self.pairings[i]
        springs.append(Spring(i1, i2, dist, k=100))

    atoms.set_constraint(springs)

    with LBFGS(atoms, maxstep=0.1, logfile=None) as opt:

        opt.run(fmax=0.05, steps=100)


    success = scramble_check(structure, atomnos, mols_graphs, max_newbonds=max_newbonds)

    return atoms.get_positions(), atoms.get_total_energy(), success

def ase_neb(reagents, products, atomnos, n_images=6, fmax=0.05, method='PM7 GEO-OK', title='temp', optimizer=LBFGS, logfile=None):
    '''
    TODO:desc
    '''

    first = Atoms(atomnos, positions=reagents)
    last = Atoms(atomnos, positions=products)

    images =  [first]
    images += [first.copy() for i in range(n_images)]
    images += [last]

    neb = DyNEB(images, fmax=fmax, climb=False,  method='eb', scale_fmax=1, allow_shared_calculator=True)
    neb.interpolate()

    dump(f'{title}_MEP_guess.xyz', images, atomnos)
    
    # Set calculators for all images
    for i, image in enumerate(images):
        image.calc = MOPAC(label='temp', command=f'{MOPAC_COMMAND} temp.mop > temp.cmdlog 2>&1', method=method)

    # Set the optimizer and optimize
    try:
        with optimizer(neb, maxstep=0.1, logfile=logfile) as opt:

            # with suppress_stdout_stderr():
            # with HiddenPrints():
            opt.run(fmax=fmax, steps=20)

            neb.climb = True
            opt.run(fmax=fmax, steps=500)

    except Exception as e:
        print(f'Stopped NEB for {title}:')
        print(e)

    energies = [image.get_total_energy() for image in images]
    ts_id = energies.index(max(energies))
    # print(f'TS structure is number {ts_id}, energy is {max(energies)}')

    os.remove(f'{title}_MEP_guess.xyz')
    dump(f'{title}_MEP.xyz', images, atomnos)
    # Save the converged MEP (minimum energy path) to an .xyz file


    return images[ts_id].get_positions(), images[ts_id].get_total_energy()

def hyperNEB(coords, atomnos, ids, constrained_indexes, reag_prod_method ='PM7', NEB_method='PM7 GEO-OK', title='temp'):
    '''
    Turn a geometry close to TS to a proper TS by getting
    reagents and products and running a climbing image NEB calculation through ASE.
    '''

    reagents = get_reagent(coords, atomnos, ids, constrained_indexes, method=reag_prod_method)
    products = get_product(coords, atomnos, ids, constrained_indexes, method=reag_prod_method)
    # get reagents and products for this reaction

    reagents -= np.mean(reagents, axis=0)
    products -= np.mean(products, axis=0)
    # centering both structures on the centroid of reactive atoms

    aligment_rotation = R.align_vectors(reagents, products)
    products = np.array([aligment_rotation @ v for v in products])
    # rotating the two structures to minimize differences

    ts_coords, ts_energy = ase_neb(reagents, products, atomnos, method=NEB_method, optimizer=LBFGS, title=title)
    # Use these structures plus the TS guess to run a NEB calculation through ASE

    return ts_coords, ts_energy

def get_product(coords, atomnos, ids, constrained_indexes, method='PM7'):
    '''
    TODO:desc
    '''

    bond_factor = 1.2
    # multiple of sum of covalent radii for two atoms.
    # If two atoms are closer than this times their sum
    # of c_radii, they are considered to converge to
    # products when their geometry is optimized. 

    step_size = 0.1
    # in Angstroms

    if len(ids) == 2:

        mol1_center = np.mean([coords[a] for a, _ in constrained_indexes], axis=0)
        mol2_center = np.mean([coords[b] for _, b in constrained_indexes], axis=0)
        motion = norm(mol2_center - mol1_center)
        # norm of the motion that, when applied to mol1,
        # superimposes its reactive atoms to the ones of mol2

        threshold_dists = [bond_factor*(pt[atomnos[a]].covalent_radius +
                                        pt[atomnos[b]].covalent_radius) for a, b in constrained_indexes]

        reactive_dists = [np.linalg.norm(coords[a] - coords[b]) for a, b in constrained_indexes]
        # distances between reactive atoms

        while not np.all([reactive_dists[i] < threshold_dists[i] for i in range(len(constrained_indexes))]):
            # print('Reactive distances are', reactive_dists)

            coords[:ids[0]] += motion*step_size

            coords, _, _ = mopac_opt(coords, atomnos, constrained_indexes, method=method)

            reactive_dists = [np.linalg.norm(coords[a] - coords[b]) for a, b in constrained_indexes]

        newcoords, _, _ = mopac_opt(coords, atomnos, method=method)
        # finally, when structures are close enough, do a free optimization to get the reaction product

        new_reactive_dists = [np.linalg.norm(newcoords[a] - newcoords[b]) for a, b in constrained_indexes]

        if np.all([new_reactive_dists[i] < threshold_dists[i] for i in range(len(constrained_indexes))]):
        # return the freely optimized structure only if the reagents did not repel each other
        # during the optimization, otherwise return the last coords, where partners were close
            return newcoords
        else:
            return coords

    else:
    # trimolecular TSs: the approach is to bring the first pair of reactive
    # atoms closer until optimization bounds the molecules together

        index_to_be_moved = constrained_indexes[0,0]
        reference = constrained_indexes[0,1]
        moving_molecule_index = next(i for i,n in enumerate(np.cumsum(ids)) if index_to_be_moved < n)
        bounds = [0] + [n+1 for n in np.cumsum(ids)]
        moving_molecule_slice = slice(bounds[moving_molecule_index], bounds[moving_molecule_index+1])
        threshold_dist = bond_factor*(pt[atomnos[constrained_indexes[0,0]]].covalent_radius +
                                      pt[atomnos[constrained_indexes[0,1]]].covalent_radius)

        motion = (coords[reference] - coords[index_to_be_moved])
        # vector from the atom to be moved to the target reactive atom

        while np.linalg.norm(motion) > threshold_dist:
        # check if the reactive atoms are sufficiently close to converge to products

            for i, atom in enumerate(coords[moving_molecule_slice]):
                dist = np.linalg.norm(atom - coords[index_to_be_moved])
                # for any atom in the molecule, distance from the reactive atom

                atom_step = step_size*np.exp(-0.5*dist)
                coords[moving_molecule_slice][i] += norm(motion)*atom_step
                # the more they are close, the more they are moved

            # print('Reactive dist -', np.linalg.norm(motion))
            coords, _, _ = mopac_opt(coords, atomnos, constrained_indexes, method=method)
            # when all atoms are moved, optimize the geometry with the previous constraints

            motion = (coords[reference] - coords[index_to_be_moved])

        newcoords, _, _ = mopac_opt(coords, atomnos, method=method)
        # finally, when structures are close enough, do a free optimization to get the reaction product

        new_reactive_dist = np.linalg.norm(newcoords[constrained_indexes[0,0]] - newcoords[constrained_indexes[0,0]])

        if new_reactive_dist < threshold_dist:
        # return the freely optimized structure only if the reagents did not repel each other
        # during the optimization, otherwise return the last coords, where partners were close
            return newcoords
        else:
            return coords

def get_reagent(coords, atomnos, ids, constrained_indexes, method='PM7'):
    '''
    TODO:desc
    '''

    bond_factor = 1.5
    # multiple of sum of covalent radii for two atoms.
    # Putting reactive atoms at this times their bonding
    # distance and performing a constrained optimization
    # is the way to get a good guess for reagents structure. 

    if len(ids) == 2:

        mol1_center = np.mean([coords[a] for a, _ in constrained_indexes], axis=0)
        mol2_center = np.mean([coords[b] for _, b in constrained_indexes], axis=0)
        motion = norm(mol2_center - mol1_center)
        # norm of the motion that, when applied to mol1,
        # superimposes its reactive centers to the ones of mol2

        threshold_dists = [bond_factor*(pt[atomnos[a]].covalent_radius + pt[atomnos[b]].covalent_radius) for a, b in constrained_indexes]

        reactive_dists = [np.linalg.norm(coords[a] - coords[b]) for a, b in constrained_indexes]
        # distances between reactive atoms

        coords[:ids[0]] -= norm(motion)*(np.mean(threshold_dists) - np.mean(reactive_dists))
        # move reactive atoms away from each other just enough

        coords, _, _ = mopac_opt(coords, atomnos, constrained_indexes=constrained_indexes, method=method)
        # optimize the structure but keeping the reactive atoms distanced

        return coords

    else:
    # trimolecular TSs: the approach is to bring the first pair of reactive
    # atoms apart just enough to get a good approximation for reagents

        index_to_be_moved = constrained_indexes[0,0]
        reference = constrained_indexes[0,1]
        moving_molecule_index = next(i for i,n in enumerate(np.cumsum(ids)) if index_to_be_moved < n)
        bounds = [0] + [n+1 for n in np.cumsum(ids)]
        moving_molecule_slice = slice(bounds[moving_molecule_index], bounds[moving_molecule_index+1])
        threshold_dist = bond_factor*(pt[atomnos[constrained_indexes[0,0]]].covalent_radius +
                                      pt[atomnos[constrained_indexes[0,1]]].covalent_radius)

        motion = (coords[reference] - coords[index_to_be_moved])
        # vector from the atom to be moved to the target reactive atom

        displacement = norm(motion)*(threshold_dist-np.linalg.norm(motion))
        # vector to be applied to the reactive atom to push it far just enough

        for i, atom in enumerate(coords[moving_molecule_slice]):
            dist = np.linalg.norm(atom - coords[index_to_be_moved])
            # for any atom in the molecule, distance from the reactive atom

            coords[moving_molecule_slice][i] -= displacement*np.exp(-0.5*dist)
            # the closer they are to the reactive atom, the further they are moved

        coords, _, _ = mopac_opt(coords, atomnos, constrained_indexes=np.array([constrained_indexes[0]]), method=method)
        # when all atoms are moved, optimize the geometry with only the first of the previous constraints

        newcoords, _, _ = mopac_opt(coords, atomnos, method=method)
        # finally, when structures are close enough, do a free optimization to get the reaction product

        new_reactive_dist = np.linalg.norm(newcoords[constrained_indexes[0,0]] - newcoords[constrained_indexes[0,0]])

        if new_reactive_dist > threshold_dist:
        # return the freely optimized structure only if the reagents did not approached back each other
        # during the optimization, otherwise return the last coords, where partners were further away
            return newcoords
        else:
            return coords

def get_nci(coords, atomnos, constrained_indexes, ids):
    '''
    Returns a list of guesses for intermolecular non-covalent
    interactions between molecular fragments/atoms. Used to get
    a hint of the most prominent NCIs that drive stereo/regio selectivity.
    '''
    nci = []
    print_list = []
    cum_ids = np.cumsum(ids)
    symbols = [pt[i].symbol for i in atomnos]
    constrained_indexes = constrained_indexes.ravel()

    for i1 in range(len(coords)):
    # check atomic pairs (O-H, N-H, ...)

            start_of_next_mol = cum_ids[next(i for i,n in enumerate(np.cumsum(ids)) if i1 < n)]
            # ensures that we are only taking into account intermolecular NCIs

            for i2 in range(len(coords[start_of_next_mol:])):
                i2 += start_of_next_mol

                if i1 not in constrained_indexes:
                    if i2 not in constrained_indexes:
                    # ignore atoms involved in constraints

                        s = ''.join(sorted([symbols[i1], symbols[i2]]))
                        # print(f'Checking pair {i1}/{i2}')

                        if s in nci_dict:
                            threshold, nci_type = nci_dict[s]
                            dist = np.linalg.norm(coords[i1]-coords[i2])

                            if dist < threshold:

                                print_list.append(nci_type + f' ({round(dist, 2)} A, indexes {i1}/{i2})')
                                # string to be printed in log

                                nci.append((nci_type, i1, i2))
                                # tuple to be used in identifying the NCI

    # checking group contributions (aromatic rings)

    aromatic_centers = []
    masks = []

    for mol in range(len(ids)):

        if mol == 0:
            mol_mask = slice(0, cum_ids[0])
            filler = 0
        else:
            mol_mask = slice(cum_ids[mol-1], cum_ids[mol])
            filler = cum_ids[mol-1]

        aromatics_indexes = np.array([i+filler for i, s in enumerate(symbols[mol_mask]) if s in ('C','N')])

        if len(aromatics_indexes) > 5:
        # only check for phenyls in molecules with more than 5 C/N atoms

            masks.append(list(it.combinations(aromatics_indexes, 6)))
            # all possible combinations of picking 6 C/N/O atoms from this molecule

    if len(masks) > 0:

        masks = np.concatenate(masks)

        for mask in masks:

            phenyl, center = is_phenyl(coords[mask])
            if phenyl:
                owner = next(i for i,n in enumerate(np.cumsum(ids)) if np.all(mask < n))
                # index of the molecule that owns that phenyl ring

                aromatic_centers.append((owner, center))

    # print(f'structure has {len(aromatic_centers)} phenyl rings')

    # checking phenyl-atom pairs
    for owner, center in aromatic_centers:
        for i, atom in enumerate(coords):

            if i < cum_ids[0]:
                atom_owner = 0
            else:
                atom_owner = next(i for i,n in enumerate(np.cumsum(ids)) if i < n)

            if atom_owner != owner:
            # if this atom belongs to a molecule different than the one that owns the phenyl

                s = ''.join(sorted(['Ph', symbols[i]]))
                if s in nci_dict:

                    threshold, nci_type = nci_dict[s]
                    dist = np.linalg.norm(center - atom)

                    if dist < threshold:

                        print_list.append(nci_type + f' ({round(dist, 2)} A, atom {i}/ring)')
                        # string to be printed in log

                        nci.append((nci_type, i, 'ring'))
                        # tuple to be used in identifying the NCI

    # checking phenyl-phenyl pairs
    for i, owner_center in enumerate(aromatic_centers):
        owner1, center1 = owner_center
        for owner2, center2 in aromatic_centers[i+1:]:
            if owner1 != owner2:
            # if this atom belongs to a molecule different than owner

                    threshold, nci_type = nci_dict['PhPh']
                    dist = np.linalg.norm(center1 - center2)

                    if dist < threshold:

                        print_list.append(nci_type + f' ({round(dist, 2)} A, ring/ring)')
                        # string to be printed in log

                        nci.append((nci_type, 'ring', 'ring'))
                        # tuple to be used in identifying the NCI

               

    return nci, print_list

def is_phenyl(coords):
    '''
    :params coords: six coordinates of C/N atoms
    :return tuple: bool indicating if the six atoms look like part of a
                   phenyl/naphtyl/pyridine system, coordinates for the center of that ring

    NOTE: quinones would show as aromatic: it is okay, since they can do π-stacking as well.
    '''
    for i, p in enumerate(coords):
        mask = np.array([True if j != i else False for j in range(6)], dtype=bool)
        others = coords[mask]
        if not max(np.linalg.norm(p-others, axis=1)) < 3:
            return False, None
    # if any atomic couple is more than 3 A away from each other, this is not a Ph

    threshold_delta = 1 - np.cos(10 * np.pi/180)
    flat_delta = 1 - np.abs(np.cos(dihedral(coords[[0,1,2,3]]) * np.pi/180))

    if flat_delta < threshold_delta:
        flat_delta = 1 - np.abs(np.cos(dihedral(coords[[0,1,2,3]]) * np.pi/180))
        if flat_delta < threshold_delta:
            # print('phenyl center at', np.mean(coords, axis=0))
            return True, np.mean(coords, axis=0)
    
    return False, None

class OrbitalSpring:
    '''
    ASE Custom Constraint Class
    Adds an harmonic force between a pair of orbitals, that is
    virtual points "bonded" to a given atom.
    '''
    def __init__(self, i1, i2, orb1, orb2, neighbors_of_1, neighbors_of_2, d_eq, k=100):
        self.i1, self.i2 = i1, i2
        self.orb1, self.orb2 = orb1, orb2
        self.neighbors_of_1, self.neighbors_of_2 = neighbors_of_1, neighbors_of_2
        self.d_eq = d_eq
        self.k = k

    def adjust_positions(self, atoms, newpositions):
        pass

    def adjust_forces(self, atoms, forces):

        # First, add the force between reactive atoms i1, i2

        orb_direction = self.orb2 - self.orb1
        # vector connecting orb1 to orb2

        spring_force = self.k * (np.linalg.norm(orb_direction) - self.d_eq)
        # absolute spring force (float). Positive if spring is overstretched.

        spring_force = np.clip(spring_force, -10, 10)
        # force is clipped at 10 eV/A

        force_direction1 = np.sign(spring_force) * norm(np.mean((norm(+orb_direction),
                                                                 norm(self.orb1-atoms.positions[self.i1])), axis=0))

        force_direction2 = np.sign(spring_force) * norm(np.mean((norm(-orb_direction),
                                                                 norm(self.orb2-atoms.positions[self.i2])), axis=0))

        # versors specifying the direction at which forces act, that is on the
        # bisector of the angle between vector connecting atom to orbital and
        # vector connecting the two orbitals

        forces[self.i1] += (force_direction1 * spring_force)
        forces[self.i2] += (force_direction2 * spring_force)
        # applying harmonic force to each atom, directed toward the other one

        # Now applying to neighbors the force derived by torque, scaled to half the spring force

        torque1 = np.cross(self.orb1 - atoms.positions[self.i1], forces[self.i1])
        for i in self.neighbors_of_1:
            forces[i] += norm(np.cross(torque1, atoms.positions[i] - atoms.positions[self.i1])) * spring_force/2

        torque2 = np.cross(self.orb2 - atoms.positions[self.i2], forces[self.i2])
        for i in self.neighbors_of_2:
            forces[i] += norm(np.cross(torque2, atoms.positions[i] - atoms.positions[self.i2])) * spring_force/2

def ase_bend(docker, original_mol, pivot, threshold, method='PM7', title='temp', traj=None):
    '''
    threshold: float (A)
    '''
    # from ase.io.trajectory import Trajectory
    # def orbitalized(atoms, orbitals):
    #     positions = np.concatenate((atoms.positions, orbitals))
    #     symbols = str(atoms.symbols) + 'n'*len(orbitals)
    #     return Atoms(symbols, positions=positions)

    mol = deepcopy(original_mol)

    i1, i2 = mol.reactive_indexes

    neighbors_of_1 = list([(a, b) for a, b in mol.graph.adjacency()][i1][1].keys())
    neighbors_of_1.remove(i1)

    neighbors_of_2 = list([(a, b) for a, b in mol.graph.adjacency()][i2][1].keys())
    neighbors_of_2.remove(i2)

    for p in mol.pivots:
        if p.index == pivot.index:
            active_pivot = p
            break
    
    dist = np.linalg.norm(active_pivot.pivot)

    for conf in range(len(mol.atomcoords)):

        atoms = Atoms(mol.atomnos, positions=mol.atomcoords[conf], calculator=MOPAC(label=title, command=f'{MOPAC_COMMAND} {title}.mop > {title}.cmdlog 2>&1', method=method))
        # traj = Trajectory(traj, mode='a', atoms=orbitalized(atoms, np.vstack([atom.center for atom in mol.reactive_atoms_classes_dict.values()])))

        for iteration in range(100):

            atoms.positions = mol.atomcoords[conf]

            orb_memo = {index:np.linalg.norm(atom.center[0]-atom.coord) for index, atom in mol.reactive_atoms_classes_dict.items()}

            orb1, orb2 = active_pivot.start, active_pivot.end

            c = OrbitalSpring(i1, i2, orb1, orb2, neighbors_of_1, neighbors_of_2, d_eq=threshold)
            atoms.set_constraint(c)

            opt = BFGS(atoms, maxstep=0.1, logfile=None)

            opt.run(fmax=0.5, steps=1)

            # traj.atoms = orbitalized(atoms, np.vstack([atom.center for atom in mol.reactive_atoms_classes_dict.values()]))
            # traj.write()

            mol.atomcoords[conf] = atoms.get_positions()

            # Update orbitals and get temp pivots
            for index, atom in mol.reactive_atoms_classes_dict.items():
                atom.init(mol, index, update=True, orb_dim=orb_memo[index], atomcoords_index=conf)
                # orbitals positions are calculated based on the conformer we are working on

            temp_pivots = docker._get_pivots(mol)

            for p in temp_pivots:
                if p.index == pivot.index:
                    active_pivot = p
                    break
            # print(active_pivot)

            dist = np.linalg.norm(active_pivot.pivot)
            # print(f'{iteration}. {mol.name} conf {conf}: pivot is {round(dist, 3)} (target {round(threshold, 3)})')

            if abs(dist - threshold) < 0.1:
                break

        if not molecule_check(original_mol.atomcoords[conf], mol.atomcoords[conf], mol.atomnos, max_newbonds=1):
            mol.atomcoords[conf] = original_mol.atomcoords[conf]
        # keep the bent structures only if no scrambling occurred between atoms

    # Now align the ensembles on the new reactive atoms positions

    reference, *targets = mol.atomcoords
    reference = np.array(reference)
    targets = np.array(targets)

    r = reference - np.mean(reference[mol.reactive_indexes], axis=0)
    ts = np.array([t - np.mean(t[mol.reactive_indexes], axis=0) for t in targets])

    output = []
    output.append(r)
    for target in ts:
        matrix = kabsch(r, target)
        output.append([matrix @ vector for vector in target])

    mol.atomcoords = np.array(output)

    # Update orbitals and pivots
    for index, atom in mol.reactive_atoms_classes_dict.items():
        atom.init(mol, index, update=True, orb_dim=orb_memo[index])

    docker._set_pivots(mol)

    return mol





# def write_orca(coords:np.array, atomnos:np.array, output, head='! PM3 Opt'):
#     '''
#     output is of _io.TextIOWrapper type

#     '''
#     assert atomnos.shape[0] == coords.shape[0]
#     assert coords.shape[1] == 3
#     head += '\n# Orca input from TSCoDe\n\n* xyz 0 1'
#     for i, atom in enumerate(coords):
#         head += '%s     % .6f % .6f % .6f\n' % (pt[atomnos[i]].symbol, atom[0], atom[1], atom[2])
#     head += '*'
#     output.write(head)

# def write_orca_neb(coords1, coords2, atomnos, title='temp', method='PM3'):

#     assert coords1.shape == coords2.shape
#     assert atomnos.shape[0] == coords1.shape[0]
#     assert coords1.shape[1] == 3

#     with open(f'{title}_start.xyz', 'w') as f:
#         write_xyz(coords1, atomnos, f, title)

#     with open(f'{title}_end.xyz', 'w') as f:
#         write_xyz(coords2, atomnos, f, title)

#     with open(f'{title}_neb.inp', 'w') as output:
#         head = f'! {method} NEB-TS\n\n'
#         head += f'%neb\nNEB_End_XYZFile "{title}_end.xyz"\nNimages 6\nend\n\n'
#         head += f'# Orca NEB input from TSCoDe\n\n* xyzfile 0 1 {title}_start.xyz'
#         output.write(head)