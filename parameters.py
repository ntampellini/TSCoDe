# Calculators parameters

MOPAC_COMMAND = 'mopac2016'
# command with which mopac will be called from the command line

THRESHOLD_KCAL = 1000
# energy threshold (kcal/mol) for TS retention after constrained optimization

orb_dim_dict = {
    'H Single Bond' : 0.85,
    'C Single Bond' : 1,
    'O Single Bond' : 1,
    'N Single Bond' : 1,
    'F Single Bond' : 1,
    'Cl Single Bond' : 1,
    'Br Single Bond' : 1,
    'I Single Bond' : 1,

    'C sp' : 1,

    'C sp2' : 1.1,
    'N sp2' : 1,

    'C sp3' : 1,
    'Br sp3' : 1,

    'O Ether' : 1,
    'S Ether' : 1,

    'O Ketone': 0.85,
    'S Ketone': 1,

    'N Imine' : 1,

    'Fallback' : 1
                }       
# Half-lenght of the transition state bonding distance involving a given atom

nci_dict={
    'FF':(3.5,'F-F interaction'),
    'HO':(2,'O-H hydrogen bond'),
    'HN':(2,'N-H hydrogen bond'),
    'HPh':(2.8,'H-Ar non-conventional hydrogen bond'), # taken from https://doi.org/10.1039/C1CP20404A
    'PhPh':(3.8, 'π-stacking interaction'),            # guessed from https://doi.org/10.1039/C2SC20045G
    # TODO - halogen bonds, π bonds interactions
}
# non covalent interaction threshold and types for atomic pairs