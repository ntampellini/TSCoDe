from periodictable import core, covalent_radius
from reactive_atoms_classes import *
pt = core.PeriodicTable(table="H=1")
covalent_radius.init(pt)

atom_type_dict = {
             'H1':Single(pt[1].covalent_radius/pt[6].covalent_radius),
             'C1':Single(1),
             'C2':'sp', # toroidal geometry
             'C3':Sp2(1), # double ball
             'C4':'sp3', # one big ball
             'N1':Single(pt[7].covalent_radius/pt[6].covalent_radius),
             'N2':'imine', # one ball on free side
             'N3':'amine', # one ball on free side
             'N4':'sp3',
             'O1':'ketone-like', # two balls 120° apart. Also for alkoxides, good enough
             'O2':'ether', # or alcohol, two balls 109,5° apart
             'S1':'ketone-like',
             'S2':'ether',
             'F1':Single(pt[9].covalent_radius/pt[6].covalent_radius),
             'Cl1':Single(pt[17].covalent_radius/pt[6].covalent_radius),
             'Br1':Single(pt[35].covalent_radius/pt[6].covalent_radius),
             'I1':Single(pt[53].covalent_radius/pt[6].covalent_radius),
             }
