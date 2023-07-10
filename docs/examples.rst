.. _exs:

Examples
========

This series of examples is meant to give guidance on how to perform a series of tasks
with TSCoDe, hoping that some of these examples will be very similar to the task you are
trying to automate. Play around!

For detailed descriptions of the operators and keywords present in the inputs, see :ref:`op_kw`.

Work is in progress to expand this section with more examples.

1. Trimolecular input
+++++++++++++++++++++

::

    DIST(a=2.135)

    maleimide.xyz 0a 5x
    opt> HCOOH.xyz 4x 1y
    csearch> dienamine.xyz 6a 23y

    # Free number of comment and blank lines!

    # First pairing (a) is the C-C reactive distance
    # Second and third pairings (x, y) are the
    # hydrogen bonds bridging the two partners.

    # Reactive pairings (a, b, c) will refine to the imposed values (here a=2.135 A)
    # Non-reactive pairings (x ,y ,z) will relax to an optimal value

    # opt> - structure of HCOOH.xyz will be optimized before running TSCoDe
    # csearch> - A conformational search will be performed on dienamine.xyz before running TSCoDe

.. figure:: /images/trimolecular.png
   :align: center
   :alt: Example output structure
   :width: 75%

   *Best transition state arrangement found by TSCoDe for the above trimolecular input, following imposed atom spacings and pairings*

2. Atropisomer rotation
+++++++++++++++++++++++

::

    SADDLE KCAL=10 CALC=MOPAC LEVEL=PM7
    atropisomer.xyz 1 2 9 10

    # Performs various clockwise/anticlockwise scans
    # at different accuracy for the specified dihedral
    # angle, performing a saddle point optimization on
    # each energy maxima above 10 kcal/mol from the lowest
    # energy structure. The calculator and the theory level
    # specified in the input override user default settings.

.. figure:: /images/atropo.png
   :alt: Example output structure
   :width: 75%
   :align: center
   
   *Best transition state arrangement found for the above input*
   
   
.. figure:: /images/plot.svg
   :alt: Example plot
   :width: 75%
   :align: center

   *Plot of energy as a function of the dihedral angle (part of TSCoDe output).*

3. Peptide-substrate binding mode
+++++++++++++++++++++++++++++++++

::

    RMSD=0.3
    csearch> hemiacetal.xyz 34x
    csearch_hb> peptide.xyz 39x

    # Complex binding mode between a reaction
    # intermediate (hemiacetal) and the catalyst
    # (peptide).

    # RMSD=0.3 reduces the similarity threshold to
    # retain more structures (default 0.5 or 1 A)

    # csearch> performs a complete conformational
    # search on hemiacetal.xyz (2 diastereomers,
    # total of 72 conformers)
    
    # csearch_hb> performs a partial conformational 
    # search on peptide.xyz, retaining the β-turn
    # hydrogen bond initially present. 19683 confs
    # generated, most diverse 1000 used for the 
    # embed (overridable with CONFS=n)

    # String algorithm: 5.18 M poses checked

    # Conformational augmentation of best poses
    # improves results further (performing a csearch
    # on every generated pose)

.. figure:: /images/peptide_chemdraw.png
   :alt: Input structures
   :width: 75%
   :align: center
   
   *Input structures for hemiacetal.xyz (left) and peptide.xyz (right)*
   
   
.. figure:: /images/peptide.png
   :alt: One output pose
   :width: 75%
   :align: center

   *Best pose generated for the above input. The yellow bond is the imposed interaction, dotted lines are hydrogen bonds*