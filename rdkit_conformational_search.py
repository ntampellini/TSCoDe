# Adapted from https://gist.github.com/tdudgeon/b061dc67f9d879905b50118408c30aac

from rdkit import Chem
from rdkit.Chem import AllChem, TorsionFingerprints
from rdkit.ML.Cluster import Butina

def csearch(filename, debug=False):

    def _gen_conformers(mol, numConfs=100, maxAttempts=1000, pruneRmsThresh=0.1, useExpTorsionAnglePrefs=True, useBasicKnowledge=True, enforceChirality=True):
        ids = AllChem.EmbedMultipleConfs(mol, numConfs=numConfs, maxAttempts=maxAttempts, pruneRmsThresh=pruneRmsThresh, useExpTorsionAnglePrefs=useExpTorsionAnglePrefs, useBasicKnowledge=useBasicKnowledge, enforceChirality=enforceChirality, numThreads=0)
        return list(ids)
        
    # def write_conformers_to_sdf(mol, filename, rmsClusters, conformerPropsDict, minEnergy):
    #     w = Chem.SDWriter(filename)
    #     for cluster in rmsClusters:
    #         for confId in cluster:
    #             for name in mol.GetPropNames():
    #                 mol.ClearProp(name)
    #             conformerProps = conformerPropsDict[confId]
    #             mol.SetIntProp("conformer_id", confId + 1)
    #             for key in conformerProps.keys():
    #                 mol.SetProp(key, str(conformerProps[key]))
    #             e = conformerProps["energy_abs"]
    #             if e:
    #                 mol.SetDoubleProp("energy_delta", e - minEnergy)
    #             w.write(mol, confId=confId)
    #     w.flush()
    #     w.close()
        
    def _calc_energy(mol, conformerId, minimizeIts):
        ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol), confId=conformerId)
        ff.Initialize()
        ff.CalcEnergy()
        results = {}
        if minimizeIts > 0:
            results["converged"] = ff.Minimize(maxIts=minimizeIts)
        results["energy_abs"] = ff.CalcEnergy()
        return results
        
    def _cluster_conformers(mol, mode="RMSD", threshold=2.0):
        if mode == "TFD":
            dmat = TorsionFingerprints.GetTFDMatrix(mol)
        else:
            dmat = AllChem.GetConformerRMSMatrix(mol, prealigned=False)
        rms_clusters = Butina.ClusterData(dmat, mol.GetNumConformers(), threshold, isDistData=True, reordering=True)
        return rms_clusters
        
    def _align_conformers(mol, clust_ids):
        rmslist = []
        AllChem.AlignMolConformers(mol, confIds=clust_ids, RMSlist=rmslist)
        return rmslist
            
        
    input_file = filename
    numConfs = 100
    maxAttempts = 1000
    pruneRmsThresh = 0.1
    clusterMethod = "RMSD"
    clusterThreshold = 2.0
    minimizeIterations = 0

    suppl = Chem.ForwardSDMolSupplier(input_file)
    i=0
    mol = [m for m in suppl][0]
    i = i+1
    m = Chem.AddHs(mol)
    # generate the confomers
    conformerIds = _gen_conformers(m, numConfs, maxAttempts, pruneRmsThresh, True, True, True)
    conformerPropsDict = {}
    for conformerId in conformerIds:
        # energy minimise (optional) and energy calculation
        props = _calc_energy(m, conformerId, minimizeIterations)
        conformerPropsDict[conformerId] = props
    # cluster the conformers
    rmsClusters = _cluster_conformers(m, clusterMethod, clusterThreshold)

    if debug: print("Molecule", i, ": generated", len(conformerIds), "conformers and", len(rmsClusters), "clusters")
    rmsClustersPerCluster = []
    clusterNumber = 0
    minEnergy = 9999999999999
    energies = []
    for cluster in rmsClusters:
        clusterNumber = clusterNumber+1
        rmsWithinCluster = _align_conformers(m, cluster)
        for conformerId in cluster:
            e = props["energy_abs"]
            if e < minEnergy:
                minEnergy = e
            energies.append(e)
            props = conformerPropsDict[conformerId]
            props["cluster_no"] = clusterNumber
            props["cluster_centroid"] = cluster[0] + 1
            idx = cluster.index(conformerId)
            if idx > 0:
                props["rms_to_centroid"] = rmsWithinCluster[idx-1]
            else:
                props["rms_to_centroid"] = 0.0
    
    number_of_clusters = max([d['cluster_no'] for d in conformerPropsDict.values()])
    good_dict = {i:{'mol':None, 'energy_abs':10000000000000} for i in range(1, number_of_clusters+1)}
    structures = m.GetConformers()
    for index, conf in conformerPropsDict.items():
        if conf['energy_abs'] < good_dict[conf['cluster_no']]['energy_abs']:
            good_dict[conf['cluster_no']]['mol'] = structures[index]
            good_dict[conf['cluster_no']]['energy_abs'] = energies[index]
    
    structures = [good_dict[i]['mol'] for i in range(1, number_of_clusters+1)]
    energies = [good_dict[i]['energy_abs'] for i in range(1, number_of_clusters+1)]

    m.RemoveAllConformers()
    for c in structures:
        m.AddConformer(c, assignId=True)
    # print('len m is', len(m.GetConformers()))
    # print('len energies is', len(energies))

    if debug:
        return structures, energies, conformerPropsDict, good_dict

    return m, energies

if __name__ == '__main__':
    import os
    from pprint import pprint
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    os.chdir('Resources')
    structures, energies, conformerPropsDict, good_dict = csearch('funky_single.sdf', debug=True)

    pprint(conformerPropsDict)
    print()
    pprint(good_dict)
    print()
    pprint(structures)
    print()
    pprint(energies)
         
