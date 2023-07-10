## TSCoDe Changelog

### 0.0.1 (Aug 10 2021)
- First release

### 0.0.2 (Aug 10 2021)
- If pivots decrease during a bend, an exception is raised. Future versions might have a different behavior in this scenario.

### 0.0.3 (Aug 10 2021)
- setup.py bugfixes.

### 0.0.5 (Sep 1 2021)
- SADDLE keyword implementation.
- Added keywords print at top of log
- Pairings are now of two types: reactive atoms (a, b, c) or NCIs (x, y, z). The latter are adjusted when specifying distances with DIST but are left free to reach their equilibrium distance (HalfSpring constraint + additional relax).
- Major code cleaning, refactoring and reordering
- Added solvent support for calculators (SOLVENT keyword)
- Dihedral embeds now support both the SADDLE and NEB keywords
- Similar structures are now pruned in a rational way: the best looking is kept (fast_score)

### 0.0.6 (Oct 19 2021)
- Updated module call adding __main__.py file to tscode/
- All internal imports are now relative to main module (import 'tscode.module' instead of 'import module')
- Maximum number of conformers per molecule is now 10, since the compenetration/similarity pruning algs are faster and more efficient (still overridden by LET keyword)
- Removed strict versioning for python required libraries
- Removed MMFF keyword
- Added NOEMBED, FFOPT, FFCALC, FFLEVEL and TS keywords
- Added a Clustered Conformational Search implementation
- Old csearch> operator is now called confab>
- New csearch> operator is the TSCoDe conformational search engine
- General code optimizations
- Added profiling command line keyword (-p)

### 0.0.7 (Oct 20 2021)
- Import bugfix

## 0.0.8 (Oct 21 2021)
- Removed unnecessary for loop in dihedral embed NEB optimizations
- Added pre-optimization before dihedral embed
- Customized text can be inserted in write_structures function
- Dihedral embed structures energies are now relative to equilibrim geometry (direct barrier height)
- Moved "test" folder inside "tscode" (fixes bug)

## 0.0.9 (Dec 2021)
- Added error message if molecule reading fails
- Operators now support spacing after the > sign
- csearch> operator now works for molecules with more than one conformer
- Improved the speed of the align_vector_pair and rot_mat_from vec functions (numba)
- Periodic table is now in pt.py, removing any cyclical import issue
- Graph manipulations are now organized in their own file, allowing shared use (graph_manipulations.py)
- The old Docker is now called Embedder, leaving the "Docker" name for future docking extension
- If no embed is recognized after applying operators, run is cleanly terminated (embedder.embed attribute in RunEmbedding.run function, run.py)
- NOEMBED keyword still works, but structure pruning now can also be performed calling the prune> operator
- conformational search now discard symmetric rotations involving 6-membered aromatic rings like phenyl, 4-pyridyl, mesityl, ...
- procs == None bugfix
- secondary amides are now considered rotable by the csearch algorithm

## 0.1.0 (May 14 2022)
- Refined conformational search - better parameters, torsion printout, fixed HB bugs, added secondary amides as rotable, made it faster and more scalable (random sampling overrides KMeans for n>1k)
- Various small bug fixes and print refinements in dihedral embed and ase_neb functions
- Added csearch_hb> operator that allows to keep the current hydrogen bonding situation in conformational sampling
- Added approach> operator that performs automatic linear scans (ideally for locating approach energy maxima)
- Torsional clustering: changed clustering method to DBSCAN over KMeans (better for spatial clusters)
- Added numerous temporary "print" statements to asses code activity at all times
- CSearch now warns when it is fed non-connected complexes of two or more molecules, if no intramolecular HBonds are found
- CSearch now has a random dihedral variant (mode 2) used for faster conformational augmentation
- Implemented conformational augmentation of TS candidates (random dihedral rotations, for three cycles, max 1000 new conformers generated)
- Refined molecule reactive atoms/pairings print statements in log
- Fully implemented the use of constrained indexes in prune>/NOEMBED runs (pairing distance can be specified now)
- Maximum number of conformers to use in embed can be specified with the CONFS keyword
- Removed enantiomers pruning for performance reasons
- RIGID keyword is automatically added for cyclical embeds with >100 conformers (override with LET)
- The NOEMBED keyword is now called REFINE, and the prune> operator is called refine>

## 0.2.0 (Jun 26 2022)
- Implemented pka> operator (xtb calculator only)
- Solved random_csearch bug (when confs < n_out)
- Refined distance constraining for OB optimizations, guided by the target length. More accurate distances, so more accurate UFF energies for constrained structures.
- Reincluded symmetrical aryl ring rotations in clustered_csearch (previously considered dummy and skipped)
- Corrected bug in the approach> operator
- Now multiple molecules are supported for approach> runs and a cumulative output is provided (distance-energy graph)
- Random Csearch now always tries to output n_out conformers (1000 max tries), instead of generating n_out confs and discarding compenetrated ones
- Implemented rsearch> operator
- Added the -cl (command line) argument, to call TSCoDe without writing an inputfile
- Added quote from database at the end of each successful run, courtesy of https://type.fit/api/quotes
- Upgraded startup banner to a benzene-like logo
- Added error logging to file via the logging module
- Solved Profiler bug (adapted to new Embedder architecture)
- Added logging for the exit status on any call to optimize(...) (supporting FF calculations too)
- Added the tscode_procs option for embedder.options, to run python code in parallel (for now just csearch_aug implemented)

## 0.3.0 (Aug 31 2022)
- Fixed run> operator bug (leftover refine> references)
- Implemented a fast, preliminary torsion fingerprint deviation-based similarity pruning (prune_confs_tfd, faster than prune_confs_rmsd) for similarity_refining
- Added this TFD pruning in the most_diverse_conformers function (this is fast enough, RMSD-based was not) and at the end of clustered_csearch and csearch_aug as well
- Refined distance constraining for XTB optimizations as well, guided by the target length. More accurate distances, so more accurate energies for constrained structures. Also less work for the final refinement step.
- Implemented wider compatibility for internal constraints - intramolecular distances that have to be respected (so that csearch is aware of them) and even enforced to a specfic distance (with DIST)

## 0.3.1 (2022)
- Fixed TFD pruning bug for embeds with no rotable bonds.
- approach> operator is now called scan> and automatically infers the scan direction (approaching or separating the atoms based on their distance).
- Fixed bug in specifying indexes without letters (in _remove_internal_constraints)
- Minor bug fixes