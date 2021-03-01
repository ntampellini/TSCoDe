# HYPERMOLECULE CLASS VARS

 'T': Temperature (K)
 'atomcoords': array, shape = (N_confs, N_atoms, 3) - atomic coordinates for each conformer
 'atomnos': array, shape = (N_atoms,) - atomic numbers
 'atoms': array, same as atomcoords but flattened to have shape = (N,3)
 'centers': array, shape = (N,3) - coordinates of orbital centers. sp2 orbitals contribute with two points, single bond orbitals with one.
 'debug': bool, prints logging messages
 'dimensions': (3,) tuple, specifying the size of box containing all atoms, in Angstroms
 'energies': array, list of read or computed relative energies for the ensemble
 'graph': <networkx.classes.graph.Graph object>, describing molecular connectivity
 'hypermolecule': array, shape = (N,3) with all coordinates of hypermolecule atoms
 'hypermolecule_atomnos': list, just like atomnos but for hypermolecule
 'name': str, relative path to file
 'orb_vers': array, shape = (N,3) with all orbital versors, for alignment purposes. sp2 orbitals contribute with two versors, single bond orbitals with one. Versors start at reactive atom and end at the relative self.center point.
 'position': array, shape = (3,) - position of molecular center, in Angstroms, used to place molecule in a TS
 'reactive_atoms_classes': list of classes, one for each reactive atom index specified. Each of these objects has the attributes described below.
 'reactive_indexes': array, shape = (N,) - containing all reactive atoms indexes
 'rootname': name, but stripped of its extension
 'rotation': array, shape = (3,3), rotation matrix specifying molecule rotation. To be used in TS building.
 'weights': array, shape = (N,) specifying normalized weights of each atom in hypermolecule.

 # REACTIVE ATOM CLASS VARS

 'alignment_matrix': array(3,3) - matrix aligning the current reactive atom orbitals with the x axis.
 'center': array(N,3) - vectors specifying the absolute position of ball orbitals for the current reactive atom
 'coord': array(3,) - coordinates of the reactive atom in consideration
 'name': str, relative path to file
 'neighbors_symbols': list of str with symbols of neighbor atoms
 'orb_vec': array(3,) - versor specifying the orientation of orbitals. Used to build alignment_matrix.
 'others': array(N,3) - vectors specifying absolute postition of linked neighbors. For 'Single Bond' class, array(3,)
 'vectors': array(N,3) - specifying vectors connecting reactive atom to linked neighbors

 Calling str(self) will return its type. ['Single Bond', 'sp2', 'sp3', ...]