.. _introduction:

Introduction
============

.. figure:: /images/logo.jpg
   :alt: TSCoDe logo
   :align: center
   :width: 600px

   When transition state embedding is a problem, TSCoDe is the solution.


.. figure:: /images/intro_embed.PNG
   :alt: Embed logic scheme
   :align: center
   :width: 700px

   *No idea how your bound structure looks like? Ask TSCoDe.*


What it is
----------

TSCoDe is a systematical conformational embedder for small molecules.
It helps computational chemists build transition states approximations and binding poses
precisely and in an automated way. It is thought as a tool to explore complex
multimolecular conformational space fast and systematically, and yield a
series of starting points for higher-level calculations.

Since its inclusion of many subroutines and functionality, it also serves as a computational toolbox
to automate various routine tasks, via either MM, semiempirical or DFT methods.

TSCoDe is written in pure Python. It leverages the Numpy and Numba libraries to perform the linear
algebra required to translate and rotate molecules and the `ASE <https://github.com/rosswhitfield/ase>`__
environment to perform a set of structure manipulations. It supports various
:ref:`external calculators <installation>` (at least one required):

-  Openbabel (>=3.1.0) (required)
-  XTB (>=6.3) (recommended)
-  ORCA (>=4.2)
-  Gaussian (>=9)
-  MOPAC2016

What it does
------------

**Generate accurately spaced poses** for bimolecular and trimolecular
transition states of organic molecules, also considering structural
deformation. If a transition state is already in hand, the distance
between reactive atoms can be specified, so as to obtain all the
stereo/regioisomeric analogs with precise molecular spacings.

Monomolecular transition states are also supported, with sigmatropic reactions in mind. (experimental)

TSCoDe is best suited for modelizations that involve many transition
state regio- and stereoisomers, where the combination of reagents
conformations is an important aspect in transition state building.

**Perform routine tasks and ensemble refinement** on conformational ensembles obtained
with other software or via the program itself, through different implementations of
conformational search algorithms.

First, :ref:`operators<op_kw>` (if provided) are applied to input structures. Then, if more
than one input file is provided and the input format conforms to some embedding algorithm
(see :ref:`some examples<exs>`) a series of poses is created and then refined. It is also
possible to perform the refinement on user-provided conformational ensembles.

How the embedding works
-----------------------

Combinations of conformations of transition state molecules are arranged in space using
some basic modeling of atomic orbitals and a good dose of linear algebra.

.. figure:: /images/orbitals.png
   :align: center
   :alt: Schematic representation of orbital models used for the embeddings
   :width: 85%

   *Schematic representation of orbital models used for the embeddings*


How the ensemble refinement works
---------------------------------

Ensemble refinement starts with a similarity pruning, evaluated through a sequence of:

 - RMSD pruning

 - TFD (torsion fingerprint deviation) pruning

 - Rotationally-corrected RMSD pruning (invariant for periodic rotation of locally symmetrical known groups, i.e. tBu, Ph)

 - Moment of Inertia along the principal axes pruning (helps remove enantiomers and rotamers along unknown locally symmetrical groups)

Extra features
--------------

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