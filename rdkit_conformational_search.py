# Adapted from https://gist.github.com/tdudgeon/b061dc67f9d879905b50118408c30aac

import os
from rdkit import Chem
from rdkit.Chem import AllChem, TorsionFingerprints
from rdkit.ML.Cluster import Butina
from rdkit.Chem.Lipinski import NumRotatableBonds
from subprocess import DEVNULL, STDOUT, check_call
from copy import deepcopy
import re

def loadbar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='#'):
	percent = ('{0:.' + str(decimals) + 'f}').format(100 * (iteration/float(total)))
	filledLength = int(length * iteration // total)
	bar = fill * filledLength + '-' * (length - filledLength)
	print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
	if iteration == total:
		print()

def csearch(filename):
    '''
    Performs a randomic conformational search and returns the conformational ensemble,
    pruned based on cluster analysis. 10**(number of heavy-atoms rotable bonds) steps.

    :params filename: an input filename, must be a .sdf single molecule file
    :return old_mol: returns the input molecule as an RDKit Mol class for later use
    :return ensemble: returns the ensemble of conformers as an RDKit Mol class
    :return energies: returns a list of absolute energies after MMFF minimization (kcal/mol)
    '''
    def _gen_conformers(mol, numConfs=100, maxAttempts=1000, pruneRmsThresh=0.1, useExpTorsionAnglePrefs=True, useBasicKnowledge=True, enforceChirality=True):
        ids = AllChem.EmbedMultipleConfs(mol, numConfs=numConfs, maxAttempts=maxAttempts, pruneRmsThresh=pruneRmsThresh, useExpTorsionAnglePrefs=useExpTorsionAnglePrefs, useBasicKnowledge=useBasicKnowledge, enforceChirality=enforceChirality, numThreads=0)
        return list(ids)
             
    def _calc_energy(mol, conformerId, minimizeIts):
        ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol), confId=conformerId)
        ff.Initialize()
        ff.CalcEnergy()
        results = {}
        if minimizeIts > 0:
            results['converged'] = ff.Minimize(maxIts=minimizeIts)
        results['energy_abs'] = ff.CalcEnergy()
        return results
        
    def _cluster_conformers(mol, mode='RMSD', threshold=2.0):
        print('Clustering conformers...')
        if mode == 'TFD':
            dmat = TorsionFingerprints.GetTFDMatrix(mol)
        else:
            dmat = AllChem.GetConformerRMSMatrix(mol, prealigned=False)
        rms_clusters = Butina.ClusterData(dmat, mol.GetNumConformers(), threshold, isDistData=True, reordering=True)
        return rms_clusters
        
    def _align_conformers(mol, clust_ids):
        rmslist = []
        AllChem.AlignMolConformers(mol, confIds=clust_ids, RMSlist=rmslist)
        return rmslist
            

    if filename.split('.')[1] == 'sdf':
        input_file = filename
    else:
        input_file = filename.split('.')[0] + '.sdf'
        check_call(f'obabel {filename} -o sdf -O {input_file}'.split(), stdout=DEVNULL, stderr=STDOUT)
        
    suppl = Chem.ForwardSDMolSupplier(input_file)
    mol = [m for m in suppl][0]
    m = Chem.AddHs(mol)
    old_mol = deepcopy(mol)
       
    # numConfs = int(min(max(10**(NumRotatableBonds(mol) + 1), 1e3), 1e6))
    numConfs = 1000
    maxAttempts = 1000
    pruneRmsThresh = 0.1
    clusterMethod = 'TFD'
    clusterThreshold = 0.2
    minimizeIterations = 200
    print(f'\nInitializing CSearch: found {NumRotatableBonds(mol)} rotable bonds, generating {numConfs} structures')


    # generate the confomers
    conformerIds = _gen_conformers(m, numConfs, maxAttempts, pruneRmsThresh, True, True, True)
    conformerPropsDict = {}
    for i, conformerId in enumerate(conformerIds):
        # energy minimise (optional) and energy calculation

        loadbar(i+1, len(conformerIds), prefix='Performing MMFF minimization... ')

        props = _calc_energy(m, conformerId, minimizeIterations)
        conformerPropsDict[conformerId] = props
    # cluster the conformers
    rmsClusters = _cluster_conformers(m, clusterMethod, clusterThreshold)

    rmsClustersPerCluster = []
    clusterNumber = 0
    minEnergy = 9999999999999
    energies = []
    for cluster in rmsClusters:
        clusterNumber = clusterNumber+1
        rmsWithinCluster = _align_conformers(m, cluster)
        for conformerId in cluster:
            e = props['energy_abs']
            if e < minEnergy:
                minEnergy = e
            energies.append(e)
            props = conformerPropsDict[conformerId]
            props['cluster_no'] = clusterNumber
            props['cluster_centroid'] = cluster[0] + 1
            idx = cluster.index(conformerId)
            if idx > 0:
                props['rms_to_centroid'] = rmsWithinCluster[idx-1]
            else:
                props['rms_to_centroid'] = 0.0
    
    number_of_clusters = max([d['cluster_no'] for d in conformerPropsDict.values()])
    good_dict = {i:{'mol':None, 'energy_abs':10000000000000} for i in range(1, number_of_clusters+1)}
    structures = m.GetConformers()
    for index, conf in conformerPropsDict.items():
        if conf['energy_abs'] < good_dict[conf['cluster_no']]['energy_abs']:
            good_dict[conf['cluster_no']]['mol'] = structures[index]
            good_dict[conf['cluster_no']]['energy_abs'] = energies[index]
    
    structures = [good_dict[i]['mol'] for i in range(1, number_of_clusters+1)]
    energies = [good_dict[i]['energy_abs'] for i in range(1, number_of_clusters+1)]
    min_e = min(energies)
    energies = [e - min_e for e in energies]

    points = zip(structures, energies)
    points = sorted(points, key=lambda point: point[1])
    structures, energies = zip(*points)

    if any([e > 10 for e in energies]):
        cut = energies.index(next(e for e in energies if e > 10))
        structures = structures[:cut]
        energies = energies[:cut]

    # print(Chem.MolToSmiles(old_mol))
    if len(re.findall('C=C', Chem.MolToSmiles(old_mol))) > 0:
    # if a double bond is present, remove structures that do not match the input stereochemistry

        temp_ens = Chem.rdchem.RWMol(m)
        temp_ens = temp_ens.GetMol()
        temp_ens.RemoveAllConformers()

        for s in structures:
            temp_ens.AddConformer(s, assignId=True)


        def get_tag(mol, i=-1):
            tag = ''
            Chem.AssignStereochemistryFrom3D(mol, confId=i)
            for bond in mol.GetBonds():
                t = str(bond.GetStereo())[6:]
                if t != 'NONE':
                    tag += t
            return tag
    
        ref_tag = get_tag(old_mol)
        new_structures = []
        new_energies = []

        for i, s in enumerate(structures):
            tag = get_tag(temp_ens, i)
            if tag == ref_tag:
                new_structures.append(s)
                new_energies.append(energies[i])
            # else:
            #     print(f'Removed structure {i}, tag: {tag}, ref_tag {ref_tag}')

        structures = new_structures
        energies = new_energies

        del temp_ens

    print('Molecule', filename, ': found', len(energies), 'different conformers')

    ensemble = Chem.rdchem.RWMol(m)
    ensemble = ensemble.GetMol()
    ensemble.RemoveAllConformers()

    for s in structures:
        ensemble.AddConformer(s, assignId=True)

    outname = filename.split('.')[0] + '_ensemble.sdf'      # Writes sigle conformers to files then convert them to one ensemble.
    writer = Chem.SDWriter(outname)                         # Really not ideal but RDKit doesn't seem to be able to output the
    for i, e in enumerate(energies):                        # .xyz ensemble required by ccread, so I don't see another way.
        if e < 10:
            writer.write(ensemble, confId=i)
    writer.flush()
    writer.close()        

    xyz_outname = filename.split('.')[0] + '_ensemble.xyz' 
    check_call(f'obabel {outname} -o xyz -O {xyz_outname}'.split(), stdout=DEVNULL, stderr=STDOUT)

    if filename != input_file:
        del suppl
        os.remove(input_file)
    os.remove(outname)

    return ensemble, energies

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    os.chdir('Resources')

    import time
    t_start = time.time()
    # ensemble, energies = csearch('funky/funky.xyz')
    # ensemble, energies = csearch('indole/indole.mol')
    # ensemble, energies = csearch('SN2/amine.mol')
    # ensemble, energies = csearch('dienamine/dienamine.xyz')
    # ensemble, energies = csearch('SN2/CH3Br.mol')
    # ensemble, energies = csearch('SN2/ketone.mol')
    # ensemble, energies = csearch('SN2/MeOH.mol')
    # ensemble, energies = csearch('SN2/flex.mol')
    ensemble, energies = csearch('EZ.mol')

    t_end = time.time()
    print(f'Took {round(t_end - t_start, 3)} s')
    print('Energies are', [round(e, 5) for e in energies], ' kcal/mol\n')        
