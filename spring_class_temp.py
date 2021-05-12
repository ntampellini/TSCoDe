import numpy as np
from linalg_tools import norm
class Spring:
    '''
    ASE Custom Constraint Class
    Adds an harmonic force between a pair of atoms.
    '''
    def __init__(self, i1, i2, d_eq, k):
        self.i1, self.i2 = i1, i2
        self.d_eq = d_eq
        self.k = k

    def adjust_positions(self, atoms, newpositions):
        pass

    def adjust_forces(self, atoms, forces):

        direction = atoms.positions[self.i2] - atoms.positions[self.i1]
        # vector connecting atom1 to atom2

        spring_force = self.k * (np.linalg.norm(direction) - self.d_eq)**2
        # absolute spring force (float)

        forces[self.i1] += norm(direction) * spring_force
        forces[self.i2] -= norm(direction) * spring_force
        # applying harmonic force to each atom, directed toward the other one


def ase_adjust_spacings(self, structure, atomnos, method='PM7'):
    '''
    '''
    atoms = Atoms(atonnos, positions=structure)
    atoms.calc = MOPAC(label='temp', command=f'{MOPAC_COMMAND} temp.mop', method=method)
    
    springs = []
    for i, dist in enumerate(self.pairings_dists):
        i1, i2 = self.pairings[i]
        springs.append(Spring(i1, i2, dist, k=10))

    atoms.set_constraints(springs)
    print(atoms.constraints)
    quit()

    with LBFGS(atoms, maxstep=0.1) as opt:

        try:
            # with suppress_stdout_stderr():
            # with HiddenPrints():
            opt.run(fmax=0.05, steps=100)

        except Exception as e:
            print(e)
            quit()

    return atoms.get_positions()

def _set_custom_orbs(self, orb_string):
    '''
    Update the reactive_atoms classes with the user-specified orbital distances.
    :param orb_string: string that looks like 'a=2.345,b=3.456,c=2.22'

    '''
    self.pairings_dists = [(piece.split('=')[0], float(piece.split('=')[1])) for piece in orb_string.split(',')]
    self.pairings_dists = sorted(self.pairings_dists, key=lambda x: x[0])

    for letter, dist in self.pairings_dists:
        for index in range(len(self.objects)):
            for pairing in self.pairings_dict[index]:

    # for each pairing specified by the user, check each pairing recorded
    # in the pairing_dict on that molecule.

                if pairing[1] == letter:
                    for reactive_index, reactive_atom in self.objects[index].reactive_atoms_classes_dict.items():
                        if reactive_index == pairing[0]:
                            reactive_atom.init(self.objects[index], reactive_index, update=True, orb_dim=dist/2)
                            # self.objects[index]._update_orbs()
                            self.log(f'--> Custom distance read: modified orbital of {index+1}. {self.objects[index].name} atom {reactive_index} to {round(dist/2, 3)} A.')
                # If the letter matches, look for the correct reactive atom on that molecule. When we find the correct match,
                # set the new orbital center with imposed distance from the reactive atom. The imposed distance is half the 
                # user-specified one, as the final atomic distances will be given by two halves of this length.
        self.log()
