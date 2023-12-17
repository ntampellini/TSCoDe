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

Since its many subroutines and functionality, it also serves as a computational toolbox
to automate various routine tasks, via either MM, semiempirical or DFT methods.

TSCoDe is written in pure Python. The linear algebra module required to translate, rotate, embed and compare
conformational ensembles is mostly compiled just-in-time with `Numba <https://numba.readthedocs.io/en/stable/>`_ and
is parallelized where possible to achieve the best possible performance and scalability. The program supports
various :ref:`external calculators <installation>` (at least one required):

-  XTB (>=6.3) (recommended)
-  ORCA (>=4.2)
-  Gaussian (>=9)
-  MOPAC2016

What it does
------------

**Generate accurately spaced poses** for bimolecular and trimolecular
transition states of organic molecules. If a transition state is already in hand, the distance
between reactive atoms can be specified, so as to obtain all the
topologically different poses with precise molecular spacings.

TSCoDe is best suited for modelizations that involve many transition
state poses and activation modes, where the combination of reagents
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

 - TFD (torsion fingerprint deviation) pruning - only for monomolecular embeds/ensembles

 - Rotationally-corrected RMSD pruning - invariant for periodic rotation of locally symmetrical *known* groups, i.e. tBu, Ph

 - MOI (moment of inertia) pruning - helps remove enantiomers and rotamers along unknown locally symmetrical groups

Extra features
--------------

**Transition state searches**

TSCoDe implements routines for locating transition states, both for poses generated
through the program and as a standalone functionality. The ``SADDLE`` and ``NEB``
keywords and the ``saddle>`` and ``neb>`` operators are available:

- With ``SADDLE``, a geometry optimization to the closest energetic maxima is performed
  on the embedded structures, using the `Sella <https://github.com/zadorlab/sella>`__ library through ASE.

- With ``NEB``, a climbing image nudged elastic band (CI-NEB) transition state
  search is performed on each embedded structure. This tends to perform best with the scan> operator,
  where the initial minimum energy path is extracted from the distance or dihedral scan points.

- The ``saddle>`` and ``neb>`` operators work in the same way on user-provided structures.

See the :ref:`operators and keywords page<op_kw>` for more details on their usage.