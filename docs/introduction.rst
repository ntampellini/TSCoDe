.. _introduction:

Introduction
============

.. image:: /images/logo.jpg
   :alt: TSCoDe logo
   :align: center
   :width: 700px
   
TSCoDe is the first systematical conformational embedder for bimolecular
and trimolecular chemical reactions. It is able to generate a
comprehensive set of both regioisomeric and stereoisomeric poses for
molecular arrangements, provided the atoms that will be reacting. It
supports both open and cyclical transition states. By feeding the
program conformational ensembles, it also generates all conformations
combinations. It is thought as a tool to explore TS conformational space
in a fast and systematical way, and yield a series of starting points
for higher-level calculations.

**NOTE: structures obtained from TSCoDe are not proper transition states
(most of the times) but can be quite close. The program is intended
to yield and rank poses, not TSs. In this way, the computational chemist
can skip the error-prone phase of molecular embedding and proceed to the
most appropriate higher-level calculation step.**

TSCoDe is written in pure Python. It leverages the numpy library to do
the linear algebra required to translate and rotate molecules, the
OpenBabel software for performing force field optimization (optional)
and the `ASE <https://github.com/rosswhitfield/ase>`__ environment to
perform a set of structure manipulations through one of the :ref:`supported
calculators <installation>`.

What the program does
---------------------

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

Extra features
--------------

**Infer differential NCIs** - After the poses generation, the program
can be told to infer the non-covalent interactions (NCIs) between
molecules in the generated structures (``NCI`` keyword). If a particular
NCI is not shared by all structures, that is reported. If a particularly
strong NCI is present only in a few TSs, this function can be a handy
tool for tracing the source of selectivity in a given chemical reaction.

**Generate proper transition state structures** (semiempirical level) -
After poses generation and refinement, these structures can be used to try
to directly obtain transition state structures at the chosen theory level.
This is not a default behavior, and it is invoked by the ``BERNY`` or ``NEB``
keywords.

- With ``SADDLE``, a geometry optimization to the closest energetic maxima is performed,
  using the `Sella <https://github.com/zadorlab/sella>`__ library through ASE.

- With ``NEB``, a climbing image nudged elastic band (CI-NEB) transition state
  search is performed after inferring both reagents and products for each
  individual pose. This entire process is of course challenging to
  automate completely, and can be prone to failures. Associative
  reactions, where two (or three) molecules are bound together (or
  strongly interacting) after the TS, with no additional species involved,
  tend to give good results. For example, cycloaddition reactions are
  great candidates while atom transfer reactions (*i.e.* epoxidations) are
  not. This is intended more as an experimenting tool than a reliable,
  definitive feature.
