.. op_kw:

Operators and keywords
======================

Operators
+++++++++

Molecule files can be preceded by *operators*, like
``opt> molecule.xyz``. They operate on the input file before it is
fed to TSCoDe embeddings, or modify the behavior of the program to
use some of its functionalities, without running a full embedding.

-  ``opt>`` - Performs an optimization of all conformers of a molecule before
   running TSCoDe. Generates a new ``molecule_opt.xyz`` file with the optimized
   coordinates.

-  ``csearch>`` - Performs a diversity-based, torsionally-clustered conformational
   search through TSCoDe. Then, an appropriate amount of the most diverse
   conformers are used to run TSCoDe. Generates a new ``molecule_confs.xyz``
   file with all optimized conformers.

-  ``confab>`` - *(deprecated)* Performs a simple confab conformational search through
   Openbabel and optimizes all obtained conformers. Then, a maximum of
   10 best conformers are used to run TSCoDe (overriden with ``LET`` keyword).
   Generates a new ``molecule_confab.xyz`` file with all optimized
   conformers. (max 7-8 rotable bonds ideally)

-  ``neb>`` - Allows the use of the TSCoDe NEB procedure on external structures, useful 
   if working with calculators that do not natively integrate such methods (*i.e.* Gaussian). 
   The implementation is a climbing image nudged elastic band (CI-NEB) TS search with 6 moving images.  
   The operator should be used on a file that contains two structures, the start and end points
   for the search. A graph with the energy of each image is written, along with the MEP guess 
   and the converged MEP.

-  ``prune>`` - Reads the (multimolecular) input file and treats it as an ensemble generated
   during a TSCoDe embedding. That is the ensemble is pruned removing similar structure, optimized
   at the theory level(s) chosen and again pruned for similarity.

Keywords
++++++++

Keywords are divided by at least one blank space. Some of them are
self-sufficient (*i.e.* ``NCI``), while some others require an
additional input (*i.e.* ``STEPS=10`` or ``DIST(a=1.8,b=2,c=1.34)``). In
the latter case, whitespaces inside the parenthesis are NOT allowed.
Floating point numbers are to be expressed with points like ``3.14``,
while commas are only used to divide keyword arguments where more than
one is accepted, like in ``DIST``.

-  **BYPASS** - Debug keyword. Used to skip all pruning steps and
   directly output all the embedded geometries.

-  **CALC** - Overrides default calculator in ``settings.py``.
   Syntax: ``CALC=ORCA``

-  **CHECK** - Visualize the input molecules through the ASE GUI, to
   check orbital positions or conformers reading faults. *(not available
   from CLI)*

-  **CLASHES** - Manually specify the max number of clashes and/or
   the distance threshold at which two atoms are considered clashing.
   The more forgiving, the more structures will reach the geometry
   optimization step. Syntax: ``CLASHES(num=3,dist=1.2)``

-  **DEBUG** - Outputs more intermediate files and information in general.
   Structural adjustments, distance refining and similar processes will
   output ASE ".traj" trajectory files. It will also produce
   "hypermolecule" ``.xyz`` files for the first conformer of each
   structure, with dummy atoms (X) in each TSCoDe "orbital" position.

-  **DEEP** - Performs a deeper search, retaining more starting
   points for calculations and smaller turning angles. Equivalent to
   ``THRESH=0.3 STEPS=24 CLASHES=(num=3,dist=1.2)``. **Use with care!**

-  **DIST** - Manually imposed distance between specified atom
   pairs, in Angstroms. Syntax uses parenthesis and commas:
   ``DIST(a=2.345,b=3.67,c=2.1)``

-  **ENANTIOMERS** - Do not discard enantiomeric structures.

-  **EZPROT** - Preserve the E or Z configuration of double bonds
   (C=C and C=N) during the embed. Likely to be useful only for
   monomolecular embeds, where molecular distortion is often great, and
   undesired isomerization processes can occur.

-  **FFCALC** - Overrides default force field calculator in ``settings.py``.
   Syntax: ``FFCALC=OB``

-  **FFLEVEL** - Manually set the theory level to be used for force field
   calculations. Default is UFF for Openbabel and Gaussian, GFN-FF for XTB.
   Standard values can be modified by running the module with the -s flag
   (recommended way, run >>>python -m tscode -s) or by manually modifying
   ``settings.py`` (not recommended).

-  **FFOPT** - Manually turn on ``FF=ON`` or off ``FF=OFF`` the force
   field optimization step, overriding the value in ``settings.py``.

-  **KCAL** - Dihedral embed: when looking for energy maxima scan
   points in order to run berny optimization, ignore scan peaks below
   this threshold value (default is 5 kcal/mol). All other embeds: trim
   output structures to a given value of relative energy (default is
   None). Syntax: ``KCAL=n``, where n can be an integer or float.

-  **LET** - Overrides safety checks that prevent the program from
   running too large calculations. Also, removes the limit of ten
   conformers per molecule in cyclical embeds.

-  **LEVEL** - Manually set the theory level to be used. Default is
   PM7 for MOPAC, PM3 for ORCA, PM6 for Gaussian and GFN2-xTB for XTB.
   White spaces, if needed, can be expressed with underscores. Be careful
   to use the syntax of your calculator, as ORCA wants a space between method
   and basis set while Gaussian a forward slash. Syntax:
   ``LEVEL(B3LYP_def2-TZVP)``. Standard values can be modified by running the
   module with the -s flag (recommended way, run >>>python -m tscode -s)
   or by manually modifying ``settings.py`` (not recommended).

-  **MTD** - Augments the conformational sampling of transition
   state candidates through the `XTB metadynamics
   implementation <https://xtb-docs.readthedocs.io/en/latest/mtd.html>`__
   (XTB calculator only).

-  **NCI** - Estimate and print non-covalent interactions present in
   the generated poses.

-  **NEB** - Perform an automatical climbing image nudged elastic
   band (CI-NEB) TS search after the partial optimization step,
   inferring reagents and products for each generated TS pose. For dihedral
   embeds, that is atropisomer rotations, scan points around the energy
   maxima are used (reliable). For all other embeds, these are guessed by obtaining
   reagents and products by bonding/distancing reactive atom pairs and
   making use of different constrained optimizations (experimental). For trimolecular
   transition states with more than one reactive pairing (``a``, ``b``
   or ``c``, not ``x``, ``y`` and ``z``), only the first, non-NCI imposed
   pairing (a) is approached - *i.e.* the C-C reactive distance in the
   example above. This ``NEB`` option is only really usable for those
   reactions in which molecules are bound together (or strongly interacting)
   after the TS, with no additional species involved (co-products). 
   For example, cycloaddition reactions are great candidates while atom
   transfer reactions (*i.e.* epoxidations) are not. Of course this
   implementation is not always reliable, and it is provided more as
   an experimenting tool than a definitive feature.

-  **NEWBONDS** - Manually specify the maximum number of "new bonds"
   that a TS structure candidate can have to be retained and not to be
   considered scrambled. Default is 0. Syntax: ``NEWBONDS=0``

-  **NOEMBED** - Same as calling ``prune>`` on a multimolecular file. 
   The program does not embed structures, but uses the input ensemble
   as a starting point as if it came out of a TSCoDe embedding.

-  **NOOPT** - Skip the optimization steps, directly writing
   structures to file after compenetration and similarity pruning.
   Dihedral embeds: performs rigid scans instead of relaxed ones.

-  **ONLYREFINED** - Discard structures that do not successfully
   refine bonding distances. Set by default with the ``SHRINK`` keyword
   and for monomolecular TSs.

-  **PROCS** - Manually set the number of cores to be used in a
   parallel ORCA calculation, overriding the default value in
   ``settings.py``. Syntax: ``PROCS=32``

-  **RIGID** - Only applies to "cyclical"/"chelotropic" embeds.
   Avoid bending structures to better build TSs.

-  **ROTRANGE** - Only applies to "cyclical"/"chelotropic" embeds.
   Manually specify the rotation range to be explored around the
   structure pivot. Default is 90. Syntax: ``ROTRANGE=90``

-  **SADDLE** - After embed and refinement, optimize structures to the 
   closest first order saddle point using the `Sella <https://github.com/zadorlab/sella>`__ library through ASE.

-  **SHRINK** - Exaggerate orbital dimensions during embed, scaling
   them by a specified factor. If used as a single keyword (``SHRINK``),
   orbital dimensions are scaled by a factor of one and a half. A syntax
   like ``SHRINK=3.14`` allows for custom scaling. This scaling makes it
   easier to perform the embed without having molecules clashing one
   into the other. Then, the correct distance between reactive atom
   pairs is achieved as for standard runs by spring constraints during
   MOPAC/ORCA optimization. The larger the scaling, the more the program
   is likely to find at least some transition state poses, but the more
   time-consuming the step of distance refinement is going to be. Values
   from 1.5 to 3 are likely to do what this keyword was though for.

-  **STEPS** - Does not apply to "monomolecular" embeds. Manually
   specify the number of steps to be taken in scanning rotations. For
   "string" embeds, the range to be explored is the full 360°, and the
   default ``STEPS=24`` will perform 15° turns. For "cyclical" and
   "chelotropic" embeds, the rotation range to be explored is
   +-\ ``ROTRANGE`` degrees. Therefore the default values, equivalent to
   ``ROTRANGE=90 STEPS=9``, will perform nine 20 degrees turns.

-  **SUPRAFAC** - Only retain suprafacial orbital configurations in
   cyclical TSs. Thought for Diels-Alder and other cycloaddition
   reactions.

-  **TS** - Uses various scans/saddle algorithms to locate the TS.
   Useful for

-  **THRESH** - RMSD threshold (Angstroms) for structure pruning.
   The smaller, the more retained structures (default is 1 A). For
   particularly small structures, a value of 0.5 is better suited, and
   it is set by default for TSs with less than 50 atoms. For dihedral
   embeds, the default value is 0.2 A. Syntax: ``THRESH=n``, where n is
   a number.