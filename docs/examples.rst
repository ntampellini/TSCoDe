.. _exs:

Examples
========

This series of examples is meant to give guidance on how to perform a series of tasks
with TSCoDe, hoping that some of these examples will be very similar to the task you are
trying to automate. Play around!

For detailed descriptions of the operators and keywords present in the inputs, see :ref:`op_kw`.

Work is in progress to expand this section with more examples.

1. Generation of a 3D structure from SMILES, conformational search and refinement
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

::

   LEVEL=GFN-FF
   refine> rsearch> opt> O=C(O)[C@H](NC([C@H](NC([C@@H](C)N)=O)C)=O)C

   # This is a comment line!

   # First row sets the level of theory at the GFN-FF level via XTB.
   # If XTB is not set as the default calculator, you can specify it
   # here adding CALC=XTB in the keyword line.

   # First, the SMILES string is converted into a 3D structure
   # (H2N-Ala-Ala-Ala-OH), then operators are applied starting
   # from the inside out:

   # opt> - the generated 3D structure will be optimized

   # rsearch> - a knowledge-based, torsion-based conformational search
   # is carried out. The number of conformers generated can be adjusted
   # with CONFS=n (default is 1000)

   # refine> - takes the 1000 conformers ensemble and starts refining it
   # by discarding identical structures, optimizes it at the GFN-FF level
   # and repeats the pruning steps. For details on the pruning, see the
   # "How TSCoDe Works" section



1. Trimolecular input
+++++++++++++++++++++

::

    DIST(a=2.135)

    maleimide.xyz 0a 5x
    opt> HCOOH.xyz 4x 1y
    csearch> dienamine.xyz 6a 23y

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
    # search on peptide.xyz, retaining the hydrogen
    # bonds present in the input structure. 19683
    # conformers are generated, and the most diverse
    # 1000 are used for the embed (override with CONFS=n)

    # String algorithm: 5.18 M poses checked

.. figure:: /images/peptide_chemdraw.png
   :alt: Input structures
   :width: 75%
   :align: center
   
   *Input structures for hemiacetal.xyz (left) and peptide.xyz (right)*
   
   
.. figure:: /images/peptide.png
   :alt: One of the output poses
   :width: 75%
   :align: center

   *Best pose generated for the above input. The yellow bond is the imposed interaction, dotted lines are hydrogen bonds*

4. Complex embedding with internal and external constraints
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

::

   DIST(a=2.0, x=1.6, y=1.6) SOLVENT=ch2cl2
   rsearch> quinazolinedione.xyz 6a 14a 0x 7y
   csearch> peptide.xyz 0x 88y 19z 80z

   # Four pairings provided (a, x, y, z):

   # a - Fixed (a, b, c letters), internal to quinazolinedione
   # (green) - kept at 2.0 Å during the entire run

   # x - Interaction (x, y, z letters) - will be embedded at 1.6 Å
   # and then relaxed during the ensemble optimization steps (red)

   # y - Interaction (x, y, z letters) -  will be embedded at 1.6 Å
   # and then relaxed during the ensemble optimization steps (orange)

   # z - Interaction (x, y, z letters), internal to peptide (light blue)
   # No distance provided, will relax during optimization

   # rsearch> - random torsion-based conformational search. Note that this is
   # internal constraints-aware, and will treat the "a" pairing as a bond

   # csearch> - diversity-based torsional conformational search. As rsearch>,
   # it is constraints-aware and will treat the "z" pairing as a bond, preventing
   # the generation of peptide conformers without the "z" interaction

.. figure:: /images/complex_embed_cd.png
   :alt: Chemdraw representation of the embed pairings
   :width: 100%
   :align: center

.. figure:: /images/qz_tscode.gif
   :alt: One of the output poses
   :width: 100%
   :align: center

   *One of the poses generated for the above input. Note how fixed constraints were mantained (a=2) while interactions were relaxed (x=1.6, y=1.6, z)*
