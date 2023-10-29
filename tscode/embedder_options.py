# coding=utf-8
'''

TSCODE: Transition State Conformational Docker
Copyright (C) 2021-2023 Nicolò Tampellini

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

'''
import os

import numpy as np

from tscode.settings import (CALCULATOR, DEFAULT_FF_LEVELS, FF_CALC,
                             FF_OPT_BOOL, PROCS, THREADS)
from tscode.utils import read_xyz

# Known keywords and relative priority level:
# 1 : First to run, set some option
# 2 : Second to run, modify variables, dependence on priority 1 options
# 3 : Third to run, modify variables, dependence on priority 2 options

keywords_dict = {
            'BYPASS' : 1,         # Debug keyword. Used to skip all pruning steps and
                                # directly output all the embedded geometries.

            'CALC' : 1,           # Manually overrides the calculator in "settings.py"
            
            'CHARGE' : 1,         # Specifies charge for the embedding 

            'CHECK' : 1,          # Visualize the input molecules through the ASE GUI,
                                # to check orbital positions or reading faults.

            'CONFS' : 1,          # Maximum number of conformers generated by csearch 
                                
            'CLASHES' : 1,        # Manually specify the max number of clashes and/or
                                # the distance threshold at which two atoms are considered
                                # clashing. The more forgiving, the more structures will reach
                                # the geometry optimization step. Syntax: `CLASHES(num=3,dist=1.2)`
            
            'DEEP' : 1,           # Performs a deeper search, retaining more starting points
                                # for calculations and smaller turning angles.

            'DEBUG' : 1,          # DEBUG KEYWORD. Writes more stuff to file.
                                
            'DIST' : 2,           # Manually imposed distance between specified atom pairs,
                                # in Angstroms. Syntax uses parenthesis and commas:
                                # `DIST(a=2.345,b=3.67,c=2.1)`

            # 'ENANTIOMERS',    # Do not discard enantiomeric structures.

            'EZPROT' : 1,         # Double bond protection

            'FFOPT' : 1,          #Manually turn on ``FF=ON`` or off ``FF=OFF`` the force
                                # field optimization step, overriding the value in ``settings.py``.

            'FFCALC' : 1,          # Manually overrides the force field calculator in "settings.py"

            'FFLEVEL' : 1,        # Manually set the theory level to be used.
                                # . Syntax: `FFLEVEL=UFF

            'IMAGES' : 1,         # Number of images to be used in NEB and mep_relax> jobs

            'KCAL' : 1,           # Trim output structures to a given value of relative energy.
                                # Syntax: `KCAL=n`, where n can be an integer or float.

            'LET' : 1,            # Overrides safety checks that prevent the
                                # program from running too large calculations

            'LEVEL' : 1,          # Manually set the theory level to be used.
                                # . Syntax: `LEVEL(PM7_EPS=6.15)
                                
            'MTD' : 1,            # Run conformational augmentation through metadynamic sampling (XTB)

            'NCI' : 1,            # Estimate and print non-covalent interactions present in the generated poses.

            'NEB' : 1,            # Perform an automatical climbing image nudged elastic band (CI-NEB)
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

            'NEWBONDS' : 1,       # Manually specify the maximum number of "new bonds" that a
                                # TS structure can have to be retained and not to be considered
                                # scrambled. Default is 1. Syntax: `NEWBONDS=1`

            'NOOPT' : 1,          # Skip the optimization steps, directly writing structures to file.

            'ONLYREFINED' : 1,    # Discard structures that do not successfully refine bonding distances.

            'PKA' : 1,            # Set reference pKa for a specific compound

            'PROCS' : 1,          # Set the number of parallel cores to be used by ORCA

            'REFINE' : 1,            # Same as calling refine> on a single file

            'RIGID' : 1,          # Does not apply to "string" embeds. Avoid
                                # bending structures to better build TSs.

            'ROTRANGE' : 1,       # Does not apply to "string" embeds. Manually specify the rotation
                                # range to be explored around the structure pivot.
                                # Default is 120. Syntax: `ROTRANGE=120`

            'SADDLE' : 1,         # After embed and refinement, optimize to first order saddle points

            'SHRINK' : 1,         # Exaggerate orbital dimensions during embed, scaling them by a factor
                                # of one and a half. This makes it easier to perform the embed without
                                # having molecules clashing one another. Then, the correct distance between
                                # reactive atom pairs is achieved as for standard runs by spring constraints
                                # during MOPAC/ORCA optimization.

            'SIMPLEORBITALS' : 1, # Override the automatic orbital assignment, using "Single" type orbitals for
                                # every reactive atom

            'SOLVENT' : 1,          # set the solvation model

            'STEPS' : 1,          # Manually specify the number of steps to be taken in scanning rotations.
                                # For string embeds, the range to be explored is the full 360°, and the
                                # default `STEPS=24` will perform 15° turns. For cyclical and chelotropic
                                # embeds, the rotation range to be explored is +-`ROTRANGE` degrees.
                                # Therefore, the default value of `ROTRANGE=120 STEPS=12` will perform
                                # twelve 20 degrees turns.

            'SUPRAFAC' : 1,       # Only retain suprafacial orbital configurations in cyclical TSs.
                                # Thought for Diels-Alder and other cycloaddition reactions.

            'RMSD' : 1,           # RMSD threshold (Angstroms) for structure pruning. The smaller,
                                # the more retained structures. Default is 0.5 A.
                                # Syntax: `RMSD=n`, where n is a number.

            'TS' : 1,             # Uses various scans/saddle algorithms to locate the TS

            'THREADS' : 1,       # Change the number of maximum concurrent processes (default is 4)
}

class Truthy_struct:
    def __bool__(self):
        return True

class Options:

    def __init__(self):

        # only used by cyclical embeds, can be set here
        self.rotation_range = 45

        # Set later by the _setup() function based on embed type,
        # since it is used by both cyclical and string embeds
        self.rotation_steps = None
        
        self.rmsd = 0.25
        self.rigid = False
        self.max_confs = 1000
        
        self.max_clashes = 0
        self.clash_thresh = 1.5

        self.max_newbonds = 0

        self.optimization = True
        self.calculator = CALCULATOR
        self.theory_level = None        # set later in _calculator_setup()
        self.solvent = None
        self.charge = 0
        self.ff_opt = FF_OPT_BOOL
        self.ff_calc = FF_CALC

        if self.ff_opt:
            self.ff_level = DEFAULT_FF_LEVELS[FF_CALC]

        self.neb = False
        self.saddle = False
        self.ts = False
        self.nci = False
        self.shrink = False
        self.shrink_multiplier = 1
        self.metadynamics = False
        self.suprafacial = False
        self.simpleorbitals = False
        self.only_refined = False
        # self.keep_enantiomers = False
        self.double_bond_protection = False
        self.keep_hb = False
        self.csearch_aug = False
        self.dryrun = False
        self.checkpoint_frequency = 20

        self.fix_angles_in_deformation = False
        # Not possible to set manually through a keyword.
        # Monomolecular embeds have it on to prevent
        # scrambling, but better to leave it off for
        # less severe deformations, since convergence
        # is faster

        self.kcal_thresh = 10
        self.bypass = False
        self.debug = False
        self.let = False
        self.check_structures = False
        self.noembed = False
        # Default values, updated if _parse_input
        # finds keywords and calls _set_options

        self.operators = []
        # this list will be filled with operator strings
        # that need to be exectured before the run. i.e. ['csearch>mol.xyz']

        self.operators_dict = {}
        # Analogous dictionary that will contain the seuquences of operators for each molecule

    def __repr__(self):
        d = {var:self.__getattribute__(var) for var in dir(self) if var[0:2] != '__'}
        
        repr_if_true = (
            'bypass',
            'check_structures',
            'csearch_aug',
            'debug',
            'let',
            'metadynamics',
            'nci',
            'neb',
            'saddle',
            'ts',
            'ff_opt',
            'noembed',
            'keep_hb',
            'operators',
            'keep_hb',
            'dryrun',
            'shrink',
            'rigid',
            'suprafacial',
            'simpleorbitals',
            'fix_angles_in_deformation',
            'double_bond_protection',
        )
        
        for name in repr_if_true:
            if not d.get(name, True):
                d.pop(name)

        repr_if_not_none = (
            'kcal_thresh',
            'solvent',
        )

        for name in repr_if_not_none:
            if d[name] is None:
                d.pop(name)

        if not FF_OPT_BOOL:
            d.pop('ff_calc')

        padding = 1 + max([len(var) for var in d])

        return '\n'.join([f'{var}{" "*(padding-len(var))}: {d[var]}' for var in d])

class OptionSetter:

    def __init__(self, embedder, *args):

        embedder.kw_line = embedder.kw_line if hasattr(embedder, 'kw_line') else ''

        self.keywords = [word.split('=')[0].upper() if not '(' in word
                                else word.split('(')[0].upper()
                                for word in embedder.kw_line.split()]

        self.keywords_simple = [k.upper() for k in embedder.kw_line.split()]
        self.keywords_simple_case_sensitive = embedder.kw_line.split()
        self.embedder = embedder
        self.args = args

        if not all(k in keywords_dict.keys() for k in self.keywords):
            for k in self.keywords:
                if k not in keywords_dict.keys():
                    SyntaxError(f'Keyword {k} was not understood. Please check your syntax.')

        if self.keywords_simple:
            embedder.log('--> Parsed keywords, in order of execution:\n    ' + ' '.join(self.sorted_keywords()) + '\n')

    def refine(self, options, *args):
        if len(self.embedder.objects) > 1:
            raise SystemExit(('REFINE keyword can only be used with one multimolecular file per run, '
                             f'in .xyz format. ({len(self.embedder.objects)} files found in input)'))

        options.noembed = True

    def _refine_operator_routine(self):
        if len(self.embedder.objects) > 1:
            raise SystemExit(('The refine> operator can only be used with one multimolecular file per run, '
                             f'in .xyz format. ({len(self.embedder.objects)} files found in input)'))

        self.embedder._set_embedder_structures_from_mol()

        if self.embedder.options.rmsd is None:
            # set this only if user did not already specify a value
            self.embedder.options.rmsd = 0.25 

        self.embedder.objects[0].compute_orbitals(override='Single' if self.options.simpleorbitals else None)

    def bypass(self, options, *args):
        options.bypass = True
        options.optimization = False

    def charge(self, options, *args):
        kw = self.keywords_simple[self.keywords.index('CHARGE')]
        options.charge = int(kw.split('=')[1])

    def confs(self, options, *args):
        kw = self.keywords_simple[self.keywords.index('CONFS')]
        options.max_confs = int(kw.split('=')[1])

    def dryrun(self, options, *args):
        options.dryrun = True

    def suprafac(self, options, *args):
        options.suprafac = True

    def deep(self, options, *args):
        options.options.rmsd = 0.1
        options.rotation_steps = 72
        options.max_clashes = 1
        options.clash_thresh = 1.4

    def rotrange(self, options, *args):
        kw = self.keywords_simple[self.keywords.index('ROTRANGE')]
        options.rotation_range = int(kw.split('=')[1])

    def steps(self, options, *args):
        kw = self.keywords_simple[self.keywords.index('STEPS')]
        options.custom_rotation_steps = int(kw.split('=')[1])

    def rmsd(self, options, *args):
        kw = self.keywords_simple[self.keywords.index('RMSD')]
        options.rmsd = float(kw.split('=')[1])

    def noopt(self, options, *args):
        options.optimization = False

    def ffopt(self, options, *args):
        kw = self.keywords_simple[self.keywords.index('FFOPT')]
        value = kw.split('=')[1].upper()
        if value not in ('ON', 'OFF'):
            raise SystemExit('FFOPT keyword can only have value \'ON\' or \'OFF\' (i.e. \'FFOPT=OFF\')')

        options.ff_opt = True if value == 'ON' else False

    def images(self, options, *args):
        kw = self.keywords_simple[self.keywords.index('IMAGES')]
        options.images = int(kw.split('=')[1])

    def bypass(self, options, *args):
        options.bypass = True
        options.optimization = False

    def dist(self, options, *args):
        kw = self.keywords_simple_case_sensitive[self.keywords.index('DIST')]
        orb_string = kw[5:-1].replace(' ','')
        # orb_string looks like 'a=2.345,b=3.456,c=2.22'

        embedder = args[0]
        embedder._set_custom_orbs(orb_string)

    def clashes(self, options, *args):
        kw = self.keywords_simple[self.keywords.index('CLASHES')]
        clashes_string = kw[8:-1].lower().replace(' ','')
        # clashes_string now looks like 'num=3,dist=1.2'

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
        options.neb = Truthy_struct()
        options.neb.images = 6
        options.neb.preopt = False

        kw = self.keywords_simple[self.keywords.index('NEB')]
        neb_options_string = kw[4:-1].lower().replace(' ','')
        # neb_options_string now looks like 'images=8,preopt=true' or ''

        if neb_options_string != '':
            for piece in neb_options_string.split(','):
                s = piece.split('=')
                if s[0].lower() == 'images':
                    options.neb.images = int(s[1])
                elif s[0].lower() == 'preopt':
                    if s[1].lower() == 'true':
                        options.neb.preopt = True
                else:
                    raise SyntaxError((f'Syntax error in NEB keyword -> NEB({neb_options_string}). ' +
                                        'Correct syntax looks like: NEB(images=8,preopt=true)'))
        
    def level(self, options, *args):
        kw = self.keywords_simple[self.keywords.index('LEVEL')]
        options.theory_level = kw.split('=')[1].upper().replace('_', ' ')

        options.theory_level = options.theory_level.replace('[', '(').replace(']', ')')
        # quick fix for testing: allows the use of square brackets
        # in place of round, so that the LEVEL keyword is not
        # mistaken for one with sub-arguments. To be better addressed 
        # when/if a major rewrite of the option setting happens.

    def fflevel(self, options, *args):
        kw = self.keywords_simple[self.keywords.index('FFLEVEL')]
        options.ff_level = kw.split('=')[1].upper().replace('_', ' ')

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

    def simpleorbitals(self, options, *args):
        options.simpleorbitals = True

    def kcal(self, options, *args):
        kw = self.keywords_simple[self.keywords.index('KCAL')]
        options.kcal_thresh = float(kw.split('=')[1])

    def shrink(self, options, *args):
        options.shrink = True
        kw = self.keywords_simple[self.keywords.index('SHRINK')]

        parsed = kw.split('=')
        options.shrink_multiplier = float(parsed[1]) if len(parsed) > 1 else 1.5

    # def enantiomers(self, options, *args):
    #     options.keep_enantiomers = True

    def debug(self, options, *args):
        options.debug = True

    def procs(self, options, *args):
        kw = self.keywords_simple[self.keywords.index('PROCS')]
        self.embedder.procs = int(kw.split('=')[1])

    def threads(self, options, *args):
        kw = self.keywords_simple[self.keywords.index('THREADS')]
        self.embedder.threads = int(kw.split('=')[1])

    def ezprot(self, options, *args):
        options.double_bond_protection = True

    def calc(self, options, *args):
        kw = self.keywords_simple[self.keywords.index('CALC')]
        options.calculator = kw.split('=')[1]

    def ffcalc(self, options, *args):
        kw = self.keywords_simple[self.keywords.index('FFCALC')]
        options.ff_calc = kw.split('=')[1]

    def mtd(self, options, *args):
        if options.calculator != 'XTB':
            raise SystemExit(('Metadynamics augmentation can only be run with the XTB calculator.\n'
                                'Change it in settings.py or use the CALC=XTB keyword.\n'))
        options.metadynamics = True

    def saddle(self, options, *args):
        if not options.optimization:
            raise SystemExit('SADDLE keyword can only be used if optimization is turned on. (Not compatible with NOOPT).')
        options.saddle = True

    def solvent(self, options, *args):
        from tscode.solvents import solvent_synonyms
        kw = self.keywords_simple[self.keywords.index('SOLVENT')]
        solvent = kw.split('=')[1].lower()
        options.solvent = solvent_synonyms.get(solvent, solvent)

    def pka(self, options, *args):
        kw = self.keywords_simple_case_sensitive[self.keywords.index('PKA')]
        pka_string, pka = kw.split('=')
        molname = pka_string[4:-1].replace(' ','')

        if molname in [mol.name for mol in self.embedder.objects]:
            if any([f'pka>{molname}' in op.replace(' ', '') for op in self.embedder.options.operators]):
                self.embedder.pka_ref = (molname, float(pka))
                return

        raise SyntaxError(f'{molname} must be present in the molecule lines, along with the pka> operator. Syntax: pka(mol.xyz)=n')

    def csearch(self, options, *args):
        options.csearch_aug = True

    def set_options(self):

        # self.keywords = sorted(self.keywords, key=lambda x: __keywords__.index(x))

        for kw in self.sorted_keywords():
            setter_function = getattr(self, kw.lower())
            setter_function(self.embedder.options, self.embedder, *self.args)

        if any('refine>' in op for op in self.embedder.options.operators) or self.embedder.options.noembed:
            self._refine_operator_routine()

    def sorted_keywords(self):
        '''
        Returns all the keywords sorted in the optimal execution order.
        
        '''
        return sorted(self.keywords, key=keywords_dict.get)