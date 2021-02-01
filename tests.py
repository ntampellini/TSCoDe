from openbabel import openbabel as ob
import os
import numpy as np

os.chdir(os.path.dirname(os.path.realpath(__file__)))
os.chdir('Resources')


def align_ensemble(filename):
    ext = filename.split('.')[-1]
    mol = ob.OBMol()
    mol_parser = ob.OBConversion()
    mol_parser.SetInAndOutFormats(ext, 'xyz')
    more = mol_parser.ReadFile(mol, filename)
    # constructor = ob.OBAlign(mol., targetmol)
    # constructor.SetTargetMol()
    # mol_parser.WriteFile(mol, filename.split('.')[:-1] + '_aligned.xyz')
    print(mol.NumConformers())


align_ensemble('dienamine.xyz')

quit()


name, ext = os.path.splitext(infile)
ext = ext[1:].lower()
# initialize mol parser
mol_parser = ob.OBConversion()
mol_parser.SetInAndOutFormats(ext,'smi')
# initialize class
bottch = BottchScore(verbose, debug)
# load the first molecule found in the file
mol = ob.OBMol()
more = mol_parser.ReadFile(mol, infile)
while more:
    ob.PerceiveStereo(mol)
    # score the molecule
    score=bottch.score(mol, disable_mesomeric)
    if not verbose:
        if show_counter:
            print("%d: %4.2f\t %s"% (counter,score,mol.GetTitle()))
        else:
            print("%4.2f\t %s"% (score,mol.GetTitle()))
    if save_png:
        name = mol.GetTitle()
        if name.strip()=="":
            name = "mol_%d" % counter
        mol_parser.SetOutFormat('png')
        mol_parser.WriteFile(mol, '%s_image.png' % name)
    more = mol_parser.Read(mol)
    counter+=1
