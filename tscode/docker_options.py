# coding=utf-8
'''

TSCODE: Transition State Conformational Docker
Copyright (C) 2021 Nicolò Tampellini

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

'''
from settings import OPENBABEL_OPT_BOOL, CALCULATOR

keywords_list = [
            'BYPASS',         # Debug keyword. Used to skip all pruning steps and
                                # directly output all the embedded geometries.

            'CALC',           # Manually overrides the calculator in "settings.py"
            
            'CHECK',          # Visualize the input molecules through the ASE GUI,
                                # to check orbital positions or reading faults.
                                
            'CLASHES',        # Manually specify the max number of clashes and/or
                                # the distance threshold at which two atoms are considered
                                # clashing. The more forgiving, the more structures will reach
                                # the geometry optimization step. Syntax: `CLASHES(num=3,dist=1.2)`
            
            'DEEP',           # Performs a deeper search, retaining more starting points
                                # for calculations and smaller turning angles. Equivalent to
                                # `THRESH=0.3 STEPS=12 CLASHES=(num=5,dist=1)`

            'DEBUG',          # DEBUG KEYWORD
                                
            'DIST',           # Manually imposed distance between specified atom pairs,
                                # in Angstroms. Syntax uses parenthesis and commas:
                                # `DIST(a=2.345,b=3.67,c=2.1)`

            'ENANTIOMERS',    # Do not discard enantiomeric structures.

            'EZPROT',         # Double bond protection

            'KCAL',           # Trim output structures to a given value of relative energy.
                                # Syntax: `KCAL=n`, where n can be an integer or float.
                                
            'LET',            # Overrides safety checks that prevent the
                                # program from running too large calculations,
                                # removes the limit of 5 conformers for cyclical embeds

            'LEVEL',          # Manually set the MOPAC theory level to be used,
                                # default is PM7. Syntax: `LEVEL=PM7`
                                
            'MMFF',           # Use the Merck Molecular Force Field during the
                                # OpenBabel pre-optimization (default is UFF).

            'MTD',            # Run conformational augmentation through metadynamic sampling (XTB)

            'NCI',            # Estimate and print non-covalent interactions present in the generated poses.

            'NEB',            # Perform an automatical climbing image nudged elastic band (CI-NEB)
                                # TS search after the partial optimization step, inferring reagents
                                # and products for each generated TS pose. These are guessed by
                                # approaching the reactive atoms until they are at the right distance,
                                # and then partially constrained (reagents) or free (products) optimizations
                                # are carried out to get the start and end points for a CI-NEB TS search.
                                # For trimolecular transition states, only the first imposed pairing (a) 
                                # is approached - i.e. the C-C reactive distance in the example above.
                                # This NEB option is only really usable for those reactions in which two
                                # (or three) molecules are bound together (or strongly interacting) after
                                # the TS, with no additional species involved. For example, cycloaddition
                                # reactions are great candidates while atom transfer reactions
                                # (i.e. epoxidations) are not. Of course this implementation is not
                                # always reliable, and it is provided more as an experimenting tool
                                # than a definitive feature.

            'NEWBONDS',       # Manually specify the maximum number of "new bonds" that a
                                # TS structure can have to be retained and not to be considered
                                # scrambled. Default is 1. Syntax: `NEWBONDS=1`

            'NOOPT',          # Skip the optimization steps, directly writing structures to file.

            'ONLYREFINED',    # Discard structures that do not successfully refine bonding distances.

            'PROCS',          # Set the number of parallel cores to be used by ORCA

            'RIGID',          # Does not apply to "string" embeds. Avoid
                                # bending structures to better build TSs.

            'ROTRANGE',       # Does not apply to "string" embeds. Manually specify the rotation
                                # range to be explored around the structure pivot.
                                # Default is 120. Syntax: `ROTRANGE=120`

            'SADDLE',         # After embed and refinement, optimize to first order saddle points

            'SHRINK',         # Exaggerate orbital dimensions during embed, scaling them by a factor
                                # of one and a half. This makes it easier to perform the embed without
                                # having molecules clashing one another. Then, the correct distance between
                                # reactive atom pairs is achieved as for standard runs by spring constraints
                                # during MOPAC/ORCA optimization.

            'STEPS',          # Manually specify the number of steps to be taken in scanning rotations.
                                # For string embeds, the range to be explored is the full 360°, and the
                                # default `STEPS=24` will perform 15° turns. For cyclical and chelotropic
                                # embeds, the rotation range to be explored is +-`ROTRANGE` degrees.
                                # Therefore, the default value of `ROTRANGE=120 STEPS=12` will perform
                                # twelve 20 degrees turns.

            'SUPRAFAC',       # Only retain suprafacial orbital configurations in cyclical TSs.
                                # Thought for Diels-Alder and other cycloaddition reactions.

            'THRESH',         # RMSD threshold (Angstroms) for structure pruning. The smaller,
                                # the more retained structures. Default is 0.5 A.
                                # Syntax: `THRESH=n`, where n is a number.
]

class Options:

    rotation_range = 90
    rotation_steps = None # Set later by the _setup() function, based on embed type
    pruning_thresh = None # Set later by the _setup() function, based on embed type/atom number
    rigid = False
    
    max_clashes = 0
    clash_thresh = 1.3

    max_newbonds = 0

    optimization = True
    calculator = CALCULATOR
    theory_level = None        # set later in _calculator_setup()
    procs = None               # set later in _calculator_setup()
    openbabel_opt = OPENBABEL_OPT_BOOL
    openbabel_level = 'UFF'

    neb = False
    saddle = False
    nci = False
    shrink = False
    metadynamics = False
    suprafacial = False
    only_refined = False
    keep_enantiomers = False
    double_bond_protection = False

    fix_angles_in_deformation = False
    # Not possible to set manually through a keyword.
    # Monomolecular embeds have it on to prevent
    # scrambling, but better to leave it off for
    # less severe deformations, since convergence
    # is faster

    kcal_thresh = None
    bypass = False
    debug = False
    let = False
    check_structures = False
    # Default values, updated if _parse_input
    # finds keywords and calls _set_options

    def __repr__(self):
        d = {var:self.__getattribute__(var) for var in dir(self) if var[0:2] != '__'}
        
        do_not_repr = (
            'bypass',
            'check_structures',
            'debug',
            'let',
            'metadynamics',
            'nci',
            'neb',
            'saddle',
        )
        
        for name in do_not_repr:
            d.pop(name)

        if self.kcal_thresh is None:
            d.pop('kcal_thresh')

        if not OPENBABEL_OPT_BOOL:
            d.pop('openbabel_level')

        if self.procs == 1 or self.calculator not in ('ORCA', ' GAUSSIAN'):
            d.pop('procs')

        padding = 1 + max([len(var) for var in d])

        return '\n'.join([f'{var}{" "*(padding-len(var))}: {d[var]}' for var in d])

class OptionSetter:

    def __init__(self, keyword_line, docker, *args):

        self.keywords = [word.split('=')[0].upper() if not '(' in word
                                else word.split('(')[0].upper()
                                for word in keyword_line.split()]

        self.keywords_simple = [k.upper() for k in keyword_line.split()]
        self.docker = docker
        self.args = args

        if not all(k in keywords_list for k in self.keywords):
            for k in self.keywords:
                if k not in keywords_list:
                    raise SyntaxError(f'Keyword {k} was not understood. Please check your syntax.')

            docker.log('--> Parsed keywords are:\n    ' + ' '.join(self.keywords_simple))

    def bypass(self, options, *args):
        options.bypass = True
        options.optimization = False

    def suprafac(self, options, *args):
        options.suprafac = True

    def deep(self, options, *args):
        options.options.pruning_thresh = 0.3
        options.rotation_steps = 24
        options.max_clashes = 3
        options.clash_thresh = 1.2

    def rotrange(self, options, *args):
        kw = self.keywords_simple[self.keywords.index('ROTRANGE')]
        options.rotation_range = int(kw.split('=')[1])

    def steps(self, options, *args):
        kw = self.keywords_simple[self.keywords.index('STEPS')]
        options.custom_rotation_steps = int(kw.split('=')[1])

    def thresh(self, options, *args):
        kw = self.keywords_simple[self.keywords.index('THRESH')]
        options.pruning_thresh = float(kw.split('=')[1])

    def noopt(self, options, *args):
        options.optimization = False
            
    def bypass(self, options, *args):
        options.bypass = True
        options.optimization = False

    def dist(self, options, *args):
        kw = self.keywords_simple[self.keywords.index('DIST')]
        orb_string = kw[5:-1].lower().replace(' ','')
        # orb_string looks like 'a=2.345,b=3.456,c=2.22'

        docker = args[0]
        docker._set_custom_orbs(orb_string)

    def clashes(self, options, *args):
        kw = self.keywords_simple[self.keywords.index('CLASHES')]
        clashes_string = kw[8:-1].lower().replace(' ','')
        # clashes_string looks like 'num=3,dist=1.2'

        for piece in clashes_string.split(','):
            s = piece.split('=')
            if s[0].lower() == 'num':
                options.max_clashes = int(s[1])
            elif s[0].lower() == 'dist':
                options.clash_thresh = float(s[1])
            else:
                raise SyntaxError((f'Syntax error in CLASHES keyword -> CLASHES({clashes_string}).' +
                                    'Correct syntax looks like: CLASHES(num=3,dist=1.2)'))
        
    def newbonds(self, options, *args):
        kw = self.keywords_simple[self.keywords.index('NEWBONDS')]
        options.max_newbonds = int(kw.split('=')[1])

    def neb(self, options, *args):
        options.neb = True

    def level(self, options, *args):
        kw = self.keywords_simple[self.keywords.index('LEVEL')]
        options.theory_level = kw.split('=')[1].upper().replace('_', ' ')

    def rigid(self, options, *args):
        options.rigid = True

    def nci(self, options, *args):
        options.nci = True

    def onlyrefined(self, options, *args):
        options.only_refined = True

    def let(self, options, *args):
        options.let = True

    def check(self, options, *args):
        options.check_structures = True

    def mmff(self, options, *args):
        options.openbabel_level = 'MMFF'

    def kcal(self, options, *args):
        kw = self.keywords_simple[self.keywords.index('KCAL')]
        options.kcal_thresh = float(kw.split('=')[1])

    def shrink(self, options, *args):
        options.shrink = True
        kw = self.keywords_simple[self.keywords.index('SHRINK')]

        parsed = kw.split('=')
        options.shrink_multiplier = float(parsed[1]) if len(parsed) > 1 else 1.5

    def enantiomers(self, options, *args):
        options.keep_enantiomers = True

    def debug(self, options, *args):
        options.debug = True

    def procs(self, options, *args):
        kw = self.keywords_simple[self.keywords.index('PROCS')]
        options.procs = int(kw.split('=')[1])

    def ezprot(self, options, *args):
        options.double_bond_protection = True

    def calc(self, options, *args):
        kw = self.keywords_simple[self.keywords.index('CALC')]
        options.calculator = kw.split('=')[1]

    def mtd(self, options, *args):
        if options.calculator != 'XTB':
            raise SystemExit(('Metadynamics augmentation can only be run with the XTB calculator.\n'
                                'Change it in settings.py or use the CALC=XTB keyword.\n'))
        options.metadynamics = True

    def saddle(self, options, *args):
        if not options.optimization:
            raise SystemExit(('SADDLE keyword can only be used if optimization is turned on. (Not compatible with NOOPT).'))
        options.saddle = True

    def set_options(self):

        # self.keywords = sorted(self.keywords, key=lambda x: __keywords__.index(x))

        for kw in self.keywords:
            setter_function = getattr(self, kw.lower())
            setter_function(self.docker.options, self.docker, *self.args)
