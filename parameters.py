# Calculators parameters
GAUSSIAN_COMMAND = 'g09'
MOPAC_COMMAND = 'mopac2016'
# MOPAC_COMMAND = r'C:\Program Files\MOPAC\mopac2016.exe'
THRESHOLD_KCAL = 10

# Distribution-correcting constants
# BREADTH = 2
BREADTH = 1e6

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