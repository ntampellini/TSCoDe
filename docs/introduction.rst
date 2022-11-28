.. _introduction:

Introduction
============

.. figure:: /images/logo.jpg
   :alt: TSCoDe logo
   :align: center
   :width: 500px

   Transition state embedding is a problem. TSCoDe is the solution.


.. figure:: /images/intro_embed.png
   :alt: Embed logic scheme
   :align: center
   :width: 700px

   *No idea how your bound structure looks like? Ask TSCoDe.*


What it is
----------

TSCoDe is a systematical conformational embedder for small molecules.
It helps computational chemists build transition states and binding poses
precisely in an automated way. It is thought as a tool to explore complex
multimolecular conformational space fast and systematically, and yield a
series of starting points for higher-level calculations.

Since its inclusion of many subroutines and functionality, it also serves as a computational toolbox
to automate various routine tasks, via either MM, semiempirical or DFT methods.

 **Structures obtained from TSCoDe are not transition states.
 The program is intended to yield and rank poses: it was born from the
 need to automate the slow and error-prone phase of molecular embedding
 that is carried before raising the theory level in computational projects.**

TSCoDe is written in pure Python. It leverages the Numpy and Numba libraries to perform the linear
algebra required to translate and rotate molecules and the `ASE <https://github.com/rosswhitfield/ase>`__
environment to perform a set of structure manipulations. It supports various
:ref:`external calculators <installation>` (at least one required):

-  MOPAC2016
-  ORCA (>=4.2)
-  Gaussian (>=9)
-  XTB (>=6.3)
-  Openbabel (3.1.0)

What it can do
--------------

**Generate accurately spaced poses** for bimolecular and trimolecular
transition states of organic molecules, also considering structural
deformation. If a transition state is already in hand, the distance
between reactive atoms can be specified, so as to obtain all the
stereo/regioisomeric analogs with precise molecular spacings.

Monomolecular transition states are also supported, with atropisomers
rotations and sigmatropic reactions in mind.

TSCoDe is best suited for modelizations that involve many transition
state regio- and stereoisomers, where the combination of reagents
conformations is an important aspect in transition state building. The
program can yield many chemically convincing atomic arrangements that
can be evaluated by the computational chemist, aiding them in exploring
all conformational and regio/stereochemical space. External programs are
meant to be used to refine these structures and obtain real TSs at higher
levels of theory. Nevetheless, the program can significatively speed up
and reinforce the systematicity of the transition state embedding process.   

How it works
------------

Some basic modeling and a good dose of linear algebra.
The complete program logic will be presented in a future publication.

Extra features
--------------

**Routine processes automation**

Over its development, the program has also become a collection of (semi)automated
computational tools that I personally like and use with good frequency.

The ``scan>`` operator automates finding a good spatial distance for a reactive
pair of atoms, particularly useful to impose that in later embeds or get starting points
for transition state searches (see the :ref:`operators <op_kw>` page).

Atropisomers rotation transition states are a routine operation in computational 
chemistry. Here, a convenient automation of the process is part of the monomolecular
embed (see the embeds table on the :ref:`usage <usg>` page). This makes the enantiomeric
process faster, easier and more robust, since a rigorous search routine is followed.

More functionality will be implemented in future releases - feature requests and
collaborations are encouraged.

.. **Infer differential NCIs** - After the poses generation, the program
.. can be told to infer the non-covalent interactions (NCIs) between
.. molecules in the generated structures (``NCI`` keyword). If a particular
.. NCI is not shared by all structures, that is reported. If a particularly
.. strong NCI is present only in a few TSs, this function can be a handy
.. tool for tracing the source of selectivity in a given chemical reaction.

**Transition state searches**

TSCoDe implements routines for locating transition states, both for poses generated
through the program and as a standalone functionality. The ``SADDLE`` and ``NEB``
keywords and the ``saddle>`` and ``neb>`` operators are available:

- With ``SADDLE``, a geometry optimization to the closest energetic maxima is performed
  on the embedded structures, using the `Sella <https://github.com/zadorlab/sella>`__ library through ASE.

- With ``NEB``, a climbing image nudged elastic band (CI-NEB) transition state
  search is performed on each embedded structure. This tends to perform best with atropisomer rotation embeddings,
  where start and end points are available, and do not need to be guessed like for other embeds.

- The ``saddle>`` and ``neb>`` operators work in the same way on user-provided structures.

See the :ref:`operators and keywords page<op_kw>` for more details on their usage.