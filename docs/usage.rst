.. _usg:

Usage
=====

The program can be run from terminal, with the command:

::

    python -m tscode myinput.txt -n [custom_title]

A custom name for the run can be optionally provided with the -n flag, otherwise a time
stamp will be used to name the output files.

It is also possible, for simple runs, to avoid creating an input file, and writing
instruction in a string after the ``-cl``/``--command_line`` argument:

::

    python -m tscode -cl "csearch> molecule.xyz"
    python -m tscode --command_line "csearch> molecule.xyz"

In this case, an input file will be written for you by the program.

Input formatting
----------------

The input can be any text file. The extension is arbitrary but I suggest
sticking with ``.txt``.

-  Any blank line will be ignored
-  Any line starting with ``#`` will be ignored
-  Keywords, if present, need to be on first non-blank, non-comment line

Then, molecule files are specified. A molecule line is made up of these elements, in this order:

-  An operator (optional)
-  The molecule file name (required)
-  Indices (numbers) and pairings (letters) for the molecule (optional)

An example with all three is ``opt> butadiene.xyz 6a 8b``.

TSCoDe can work with all molecular formats read by
`cclib <https://github.com/cclib/cclib>`__, but best practice is using
only the ``.xyz`` file format, particularly for multimolecular files
containing different conformers of the same molecule. **Molecule indices
are counted starting from zero!**

Operators
+++++++++

The first step of every run is the execution of the requested operators. See the
:ref:`operators <op_kw>` page to see the full set of tools available.

Embedding runs
++++++++++++++

Then, if the input you provided is consistent with an embedding, one will be carried out.
Embeddings can be of six kinds:

-  **monomolecular** - One molecule, two reactive atoms (*i.e.*
   Cope rearrangements)
-  **dihedral** - One molecule, four reactive atoms (*i.e.*
   racemization of BINOL)
-  **string** - Two molecules, one reactive atom each (*i.e.* SN2
   reactions)
-  **chelotropic** - Two molecules, one with a single reactive
   atom and the other with two reactive atoms (*i.e.* epoxidations)
-  **cyclical** (bimolecular) - Two molecules, two reactive atoms
   each (*i.e.* Diels-Alder reactions)
-  **cyclical** (trimolecular) - Three molecules, two reactive
   atoms each (*i.e.* reactions where two partners are bridged by a
   carboxylic acid like the example above)

Reactive atoms supported include various hybridations of
``C, H, O, N, P, S, F, Cl, Br and I``. Many common metals are also
included (``Li, Na, Mg, K, Ca, Ti, Rb, Sr, Cs, Ba, Zn``), and it is easy
to add more if you need them (from *reactive_atoms_classes.py*). 

.. figure:: /images/embeds.svg
   :alt: Embeds Infographic
   :align: center
   :width: 700px

   *Colored dots represent imposed atom pairings. Note that monomolecular embeds only support two reactive atoms at the moment (feature requests are encouraged).*

Pairings
++++++++

After each reactive index, it is possible to specify a pairing letter (``a``,
``b`` or ``c``) to represent the "flag" of that atom. If provided, the
program will only yield the poses that respect these atom
pairings. It is also possible to specify more than one flag per atom,
useful for chelotropic embeds - *i.e.* the hydroxyl oxygen atom of a peracid, as
``4ab``.

.. If a ``NEB`` calculation is to be performed on a trimolecular transition
.. state, the reactive distance "scanned" is the first imposed (``a``). See
.. ``NEB`` keyword in the keyword section.

Good practice and suggested options (work in progress)
------------------------------------------------------

When modeling a reaction through TSCoDe, I suggest following these
guidelines. Not all of them apply to all embed types, but they will
surely help in leveraging the program in the best way.

1) Assess that the reaction is supported by TSCoDe, and plan on what the
input will look like. See Input formatting above for help.

1) Obtain molecular structures in .xyz format. If more conformers are provided,
they must be in a multimolecular ``.xyz`` file, and **atom ordering
must be consistent throughout all structures.** Otherwise, they will just be
skipped by the module used to read molecular files (cclib).

2) If a given molecule is present in the transition state, but it is
not strictly involved in bonds breaking/forming, then that molecule
needs to be pre-complexed to the one with which it is interacting. That is,
the bimolecular complex should be used. This can be the case for non-covalent
catalysts. For example, if the reaction between a ketone and a metal enolate
is catalyzed by a thiourea that activates the ketone carbonyl group, then the
TSCoDe modelization of the reaction should be bimolecular. The first molecule
is the ketone-thiourea interaction complex while the second one is the metal enolate.

3) Make sure to use the ``csearch>`` and/or ``csearch_hb>`` operators or provide conformational
ensembles obtained with other software. Note that the CSearch implementation here
is meant to be fast, scalable, and efficient, and is not able to sample ring conformations.
   
4) Understand what atoms are reacting for each structure and record
their index (**starting from 0!**). If you are unsure of reactive atomic
indices, you can run a test input without indices, and the program will
ask you to manually specify them from the ASE GUI by clicking. This is
not possible if you are running TSCoDe on STPs with no desktop access.
When choosing this option of manually picking atoms, it is not
possible to specify atom pairings. Therefore, I suggest using this
option only to check the reactive atoms indices and then building a
standard input file.

5) Optionally, after specifying reactive indices, the ``CHECK`` keyword
can be used. A series of pop-up ASE GUI windows will be displayed,
showing each molecule with a series of red dots around the reactive
atoms chosen. This can be used to check "orbital" positions or conformer
reading faults (scroll through conformers with page-up and down
buttons). Program will terminate after the last visualization is closed.

6) I try to tweak TSCoDe default parameters to yield good results for any situation
without specifying any keyword or atom pairing. However, if you
have some information about your system, I strongly encourage you to specify all
the desired pairings and options. Trimolecular TSs without imposed pairings are 8
times more than the ones with defined pairings. Also, if you have an
accurate idea of the distances between reactive atoms in your desired
TSs, the ``DIST`` keyword can yield structures that are *very* close to
higher theory level TSs. These can come from a previous higher-level
calculation or can be inferred by similar reactions. If no pairing
distances are provided, a guess is performed based on the atom type by reading editable
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
Note that the use of ``RIGID``, while speeding up cyclical embeds
considerably, could jeopardize finding some transition state arrangements.

9) If you are not sure about what to do, or have any other questions I will be
happy to have a chat with you. Send me an email `here <mailto:nicolo.tampellini@yale.edu>`__.