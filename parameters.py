# Calculators parameters

MOPAC_COMMAND = 'mopac2016'
# command with which mopac will be called from the command line

THRESHOLD_KCAL = 1000
# energy threshold (kcal/mol) for TS retention after constrained optimization

orb_dim_dict = {
                'H Single Bond' : 1,
                'C Single Bond' : 1,
                'O Single Bond' : 1,
                'N Single Bond' : 1,
                'F Single Bond' : 1,
                'Cl Single Bond' : 1,
                'Br Single Bond' : 1,
                'I Single Bond' : 1,

                'C sp' : 1,

                'C sp2' : 1,
                'N sp2' : 1,

                'C sp3' : 1,
                'Br sp3' : 1,

                'O Ether' : 1,
                'S Ether' : 1,

                'O Ketone': 1,
                'S Ketone': 1,

                'N Imine' : 1,
                }       
# Half-lenght of the transition state bonding distance involving that atom