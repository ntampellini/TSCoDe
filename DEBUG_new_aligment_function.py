from openbabel import openbabel as ob
import os
import numpy as np


def _align_ensemble(filename, reactive_atoms):

    '''
    Align a set of conformers to the first one, writing a new ensemble file.
    Alignment is done on reactive atom(s) and its immediate neighbors.

    filename            Input file name. Can be anything, .xyz preferred
    reactive_atoms      Index of atoms that will link during the desired reaction.
                        May be either int or list of int.
    return              Writes a new filename_aligned.xyz file and returns its name

    '''

    rootname, ext = filename.split('.')
    mol_parser = ob.OBConversion()
    mol_parser.SetInFormat(ext)
    mol = ob.OBMol()
    notatend = mol_parser.ReadFile(mol, filename)
    allmols = []
    while notatend:
        allmols.append(mol)
        mol = ob.OBMol()
        notatend = mol_parser.Read(mol)
    # Crazy but this is standard Openbabel procedure. Anyway,looks like a pybel method is available
    # to doall this actually https://bmcchem.biomedcentral.com/articles/10.1186/1752-153X-2-5
    
    del mol

    # fragments = []
    # for m in allmols:
    #     fragments.append(m.Get)

    # ref = []    # Setting reference molecule to align all others, and aligning all to first
    # obatom = allmols[0].GetAtom(reactive_atoms)
    # for neighbour_atom in ob.OBAtomAtomIter(obatom):
    #     ref.append(neighbour_atom)

    ref = allmols[0]
    constructor = ob.OBAlign()
    constructor.SetRefMol(ref)
    for m in allmols[1:]:
        constructor.SetTargetMol(m)
        constructor.Align()           # TO DO: ALIGN TO REACTIVE INDEXES AND NOT RANDOMLY - NEIGHBORS OF INDEX 6 SHOULD BE 1, 7, 19
        constructor.UpdateCoords(m)

    mol_parser.SetOutFormat('.xyz')
    outlist = []
    for i, m in enumerate(allmols):
        name = rootname + f'_aligned_{i}.xyz'
        mol_parser.WriteFile(m, name)
        outlist.append(name)
    mol_parser.CloseOutFile()
    # CloseOutFile() call required to close last molecule stream since mol_parser won't be called again
    name = rootname + '_aligned.xyz'
    # mol_parser.FullConvert(outlist, name, None)
        # I really don't know why this doesn't work, i want to cry
        # http://openbabel.org/api/current/classOpenBabel_1_1OBConversion.shtml#a9d12b0f7f38951d2d1065fc7ddae4229
        # This should help on function syntax but I don't know much C++. I had to take a stupid detour from os.system :(
    os.system(f'obabel {rootname}_aligned*.xyz -o xyz -O {name}')
    for m in outlist:
        os.remove(m)
    return name


####################################################################

os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('Resources')

test = 'dienamine.xyz'
test = 'funky.xyz'


_align_ensemble(test, 6)
 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cclib.io import ccread

aligned = ccread(test.split('.')[0] + '_aligned.xyz')
# aligned = ccread('funky_correct_alignment.xyz')         #This is how it should come out

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

aligned_atoms = np.array([atom for structure in aligned.atomcoords for atom in structure])
x = [atom[0] for atom in aligned_atoms]
y = [atom[1] for atom in aligned_atoms]
z = [atom[2] for atom in aligned_atoms]

col = {
    1:'gray',
    6:'black',
    7:'blue',
    8:'red'
}
col_list = [col[i] for i in aligned.atomnos]
col_list_full = []
for s in aligned.atomcoords:
    for c in col_list:
        col_list_full.append(c)

plot = ax.scatter(x, y, z, c=col_list_full, label='aligned')
ax.legend()
plt.show()