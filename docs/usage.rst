.. _usage:

Usage
=====

The program can be run from terminal, with the command:

::

    python -m tscode myinput.txt [custom_title]

A custom title for the run can be optionally provided, otherwise a time
stamp will be used to name the output files.

Examples of ``myinput.txt``
---------------------------

Trimolecular input
++++++++++++++++++

::

    DIST(a=2.135,b=1.548,c=1.901) NEB
    maleimide.xyz 0a 5b
    opt>HCOOH.xyz 4b 1c
    csearch>dienamine.xyz 6a 23c

    # Free number of comment lines!

    # First pairing (a) is the C-C reactive distance
    # Second and third pairings (b, c) are the
    # hydrogen bonds bridging the two partners.

    # Structure of HCOOH.xyz will be optimized before running TSCoDe
    # A conformational analysis will be performed on dienamine.xyz before running TSCoDe

.. figure:: /images/trimolecular.png
   :align: center
   :alt: Example output structure
   :width: 75%

   *Best transition state arrangement found by TSCoDe for the above trimolecular input, following imposed atom spacings and pairings*

Atropisomer rotation
++++++++++++++++++++

::

    SADDLE KCAL=10 CALC=MOPAC LEVEL=PM7
    atropisomer.xyz 1 2 9 10

.. figure:: /images/atropo.png
   :alt: Example output structure
   :width: 75%
   :align: center
   
   *Best transition state arrangement found by TSCoDe for the above input*
   
   
.. figure:: /images/plot.svg
   :alt: Example plot
   :width: 75%
   :align: center

   *Plot of energy as a function of the dihedral angle (part of TSCoDe output).*

Input formatting
----------------

The input can be any text file. The extension is arbitrary but I suggest
sticking with ``.txt``.

-  Any blank line will be ignored
-  Any line starting with ``#`` will be ignored
-  Keywords, if present, need to be on first non-blank, non-comment line
-  Then, two or three molecule files are specified, along with their
   reactive atoms indexes, and eventually their pairings

TSCoDe can work with all molecular formats read by
`cclib <https://github.com/cclib/cclib>`__, but best practice is using
only the ``.xyz`` file format, particularly for multimolecular files
containing different conformers of the same molecule. **Reactive indexes
are counted starting from zero!** If the molecules are specified without
reactive indexes, a pop-up ASE GUI window will guide the user into
manually specifying the reactive atoms after running the program *(not
available from CLI)*.

Reactive atoms supported include various hybridations of
``C, H, O, N, P, S, F, Cl, Br and I``. Many common metals are also
included (``Li, Na, Mg, K, Ca, Ti, Rb, Sr, Cs, Ba, Zn``), and it is easy
to add more if you need them. Reactions can be of six kinds:

-  **monomolecular** embed - One molecule, two reactive atoms (*i.e.*
   Cope rearrangements)
-  **dihedral** embed - One molecule, four reactive atoms (*i.e.*
   racemization of BINOL)
-  **string** embed - Two molecules, one reactive atom each (*i.e.* SN2
   reactions)
-  **chelotropic** embed - Two molecules, one with a single reactive
   atom and the other with two reactive atoms (*i.e.* epoxidations)
-  **cyclical** embed (bimolecular) - Two molecules, two reactive atoms
   each (*i.e.* Diels-Alder reactions)
-  **cyclical** embed (trimolecular) - Three molecules, two reactive
   atoms each (*i.e.* reactions where two partners are bridged by a
   carboxylic acid like the example above)

.. figure:: /images/embeds.svg
   :alt: Embeds Infographic
   :align: center
   :width: 700px

   *Colored dots represent imposed atom pairings. Note that monomolecular embeds only support two reactive atoms at the moment (feature requests are encouraged).*

After each reactive index, it is possible to specify a letter (``a``,
``b`` or ``c``) to represent the "flag" of that atom. If provided, the
program will only yield the regioisomers that respect these atom
pairings. For "chelotropic embeds", one could specify that a single atom
has two flags, for example the hydroxyl oxygen atom of a peracid, like
``4ab``.

If a ``NEB`` calculation is to be performed on a trimolecular transition
state, the reactive distance "scanned" is the first imposed (``a``). See
``NEB`` keyword in the keyword section.

Operators
+++++++++

Molecule files can be preceded by *operators*, like
``opt>molecule.xyz``. They operate on the input file before it is
fed to TSCoDe. It is important not to include any space character
between the operator and the molecule name.

-  ``opt>`` - Performs an optimization of the structure(s) before
   using it/them in TSCoDe. Generates a new ``molecule_opt.xyz`` file
   with the optimized coordinates.

-  ``csearch>`` - Performs a diversity-based, torsionally-clustered conformational search through
   TSCoDe. Then, a maximum of 10 best
   conformers are used to run TSCoDe (overriden with ``LET`` keyword).
   Generates a new ``molecule_confs.xyz`` file with all optimized
   conformers.

-  ``confab>`` - Performs a simple confab conformational search through
   Openbabel and optimizes all obtained conformers. Then, a maximum of 10 best
   conformers are used to run TSCoDe (overriden with ``LET`` keyword).
   Generates a new ``molecule_confab.xyz`` file with all optimized
   conformers. (max 7-8 rotable bonds ideally)

Good practice and suggested options (work in progress)
++++++++++++++++++++++++++++++++++++++++++++++++++++++

When modeling a reaction through TSCoDe, I suggest following these
guidelines. Not all of them apply to all embed types, but they will
surely help in leveraging the program in the best way.

0) Assess that the reaction is supported by TSCoDe. See Input
formatting.

1) Obtain molecular structures in .xyz format. If more conformers are to
be used, they must be in a multimolecular ``.xyz`` file, and atom ordering
must be consistent throughout all structures.

2) If a given molecule is present in the transition state, but it is
not strictly involved in bonds breaking/forming, then that molecule
needs to be joined with the one with which it is interacting. That is,
this new molecule should be the bimolecular interaction complex. This is
often the case for catalysts. For example, if the reaction between a
ketone and a metal enolate is catalyzed by a thiourea that activates the
ketone carbonyl, then the TSCoDe modelization of the reaction should be
bimolecular. The first molecule is the ketone-thiourea interaction
complex while the second one is the metal enolate.

3) Use the ``csearch>`` operator or provide conformational
ensembles.
   
4) Understand what atoms are reacting for each structure and record
their index (**starting from 0!**). If you are unsure of reactive atomic
indexes, you can run a test input without indexes, and the program will
ask you to manually specify them from the ASE GUI by clicking. This is
not possible if you are running TSCoDe from a command line interface
(CLI). When choosing this option of manually picking atoms, it is not
possible to specify atom pairings. Therefore, I suggest using this
option only to check the reactive atoms indexes and then building a
standard input file.

5) Optionally, after specifying reactive indexes, the ``CHECK`` keyword
can be used. A series of pop-up ASE GUI windows will be displayed,
showing each molecule with a series of red dots around the reactive
atoms chosen. This can be used to check "orbital" positions or conformer
reading faults (scroll through conformers with page-up and down
buttons). Program will terminate after the last visualization is closed.

6) By default, TSCoDe parameters are optimized to yield good results
without specifying any keyword nor atom pairing. However, if you already
have information on your system, I strongly encourage you to specify all
the desired pairings. Trimolecular TSs without imposed pairings are 8
times more than the ones with defined pairings. Also, if you have an
accurate idea of the distances between reactive atoms in your desired
TSs, the ``DIST`` keyword can yield structures that are *very* close to
higher theory level TSs. These can come from a previous higher-level
calculation or can be inferred by similar reactions. If no pairing
distances are provided, a guess is performed by reading editable
parameters on the ``parameters.py`` file.

7) If the reaction involves big molecules, or if a lot of conformations
are to be used, a preliminar run using the ``NOOPT`` keyword may be a
good idea to see how many structures are generated and would require
MOPAC/ORCA optimization in a standard run.

8) If TSCoDe does not find any suitable candidate for the given reacion,
most of the times this is because of compenetration pruning. This mean
that a lot of structures are generated, but all of them have some atoms
compenetrating one into the other, and are therefore discarded. A
solution could be to loosen the compenetration rejection citeria
(``CLASHES`` keyword, not recommended) or to use the ``SHRINK`` keyword
(recommended, see keywords section). Note that ``SHRINK`` calculations
will be loger, as MOPAC/ORCA/GAUSSIAN distance-refining optimizations
through ASE will require more iterations to reach target distances.