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
- Fully implemented the use of constrained indices in prune>/NOEMBED runs (pairing distance can be specified now)
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

## 0.3.1 (Nov 27 2022)
- Fixed TFD pruning bug for embeds with no rotable bonds.
- approach> operator is now called scan> and automatically infers the scan direction (approaching or separating the atoms based on their distance).
- Fixed bug in specifying indices without letters (in _remove_internal_constraints)
- Minor bug fixes
- Added flexibility in NEB keyword, allowing optimization of start/end points and specifying the number of images (NEB keyword)
- NEB calculations now support two, three or a greater odd number of structures as input, to facilitate computational refinement of MEPs

## 0.3.2 (Nov 29 2022)
- Removed Walrus operators from the code for Python 3.7 compatibility

## 0.3.3 (July 10 2023)
- Added "autoneb>" operator, that automatically builds a MEP guess based on input structure. Currently only supporting 7-membered rings inversions.
- The Hypermolecule class can now also accept SMILES strings instead of molecular files. Only for embeds with no index to be specified, as "autoneb>"
- Changed run> operator back to refine> (and RUN keyword bask to REFINE).
- Fixed bug with scan> termination called when not required.
- Changed default for conformational search during embeds to false, and NOCSEARCH keyword to CSEARCH.
- Clustered csearch module: added flexibility in some functions to allow external use of them.

## 0.3.4 (August 30 2023)
- Added the ability to recognize interrupted refine> runs and restart from last checkpoint.
- Moreover, checkpoints now update every 20 optimized structures.
- Layered optimization protocol for optimization_refinement when using ORCA - first round with 3 iterations, then 5 extra, then to convergence.
- XTB calculations that do not reach convergence do not crash the program anymore unless specified with assert_convergence=True.
- XTB force field ensemble optimizations are carried in two steps, first with loose convergence and then with tight convergence.
- Various small bugfixes and printout beautifications.

## 0.3.5 (September 22 2023)
- Added rotationally-corrected RMSD pruning, to treat symmetrical rotations and get rid of identical rotamers that only differ from indexing order. The treatment is skipped for ensembles greater than 750 structures, to avoid unnecessary slowing down the refinement process.
- Renamed clustered_csearch.py as torsion_module.py, since it now contains mostly  torsion-based constructs and utilities.
- Resolved a bug for internal constraints preventing paired embeds to be recognized as correct.
- Implemented "multiembed" embeds, able to perform bimolecular cyclical embeds systematically on every arrangement of pairs of multiple atoms (see documentation)
- Fixed dependency requirements (again?)

## 0.3.6 (September 24 2023)
- Implemented nested operators call ('refine> rsearch> opt> mol.xyz 2a 7a'), executed in reversed order (opt>, then rsearch>, then refine>)
- Removed checkpoint automatic restart, as it was not compatible with the new nested operator routine
- To remedy the last point, the last 'checkpoint' ensemble is now not deleted after optimization_refinement
- The opt> operator is now aware of internal constraints and performs constrained optimizations (moved letter/pairing/dist functions from RunEmbedder to Embedder)
- Updated README and ReadTheDocs documentations

## 0.3.7 (September 28 2023)
- Added energy table at the end of runs that generate energetic data
- Fixed bug in force_field_refining that runs with no pairings
- Fixed lack of output in optimization_refining when less than 20 structures were optimized
- Brought back MOI-based pruning, when the ensemble is <200 structures, as it proves fast and beneficial for locally symmetric (dummy) rotations not yet identified by rmsd_rot_corr. Threshold is statically set at 1% (10E-2). In the future, it could be possible to uniform to CREST's approach of adjusting it from 1 to 2.5 % dynamically based on the anisotropy of moments of inertia.
- Adjusted defaul threshold for RMSD similarity from 0.5 to 0.25 A (benchmarked to retain all methylcyclohexane conformers)
- Cleaned residuals of the ENANTIOMERS keyword and related prune_enantiomers (now prune_by_moment_of_inertia)
- Renamed every "indexes" to "indices", as I should have done long time ago...
- The function distance_refinement is now just incorporated in force_field_refinement and optimization_refinement as an option (only_fixed_constraints). First, an optimization is done with all specified constraints (fixed and interactions, at loose convergence for force field) and then the interaction constraints are released (tight convergence for FF). This process should be more robust than the one before, as it minimizes scrambling/separation of multimolecular arrangements and avoids the limitations of xtb-python when using XTB as calculator (since everything is now dealt with without ASE). Moreover, energy pruning is only performed after the interaction constraints have been released, so that random fluctuations in interaction distances constraints do not bias conformer selection (imperfect embedding geometries in mind).
- Rotationally corrected RMSD-based pruning in similarity_refining is now only done for molecules with at least one locally symmetric torsion (saves time)

## 0.3.8 (October 5 2023)
- Expanded debug functionality (DEBUG keyword): printouts of RunEmbedding status (dump_status)
- Added a xtb_pre_opt wrapper for xtb_opt that retains every bond present in the initial set of graphs, in addition to provided constraints. This will greatly increase the quality of structures coming from embeds into force_field_refining. Added before force_field_opt for embed runs (>1 self.objects) that use XTB as FF calculator.
- Introduced orbital subtypes (atom.subtype) to offer more flexibility for embed types. These do not alter default pairing_dists but allow for specific orbital arrangements around the reactive atom.
- New Ketone atom subtype: 'trilobe' - three lobes at the opposite end of alkoxide/sulfonamide substituents relative to oxygen.
- Added the DRYRUN keyword. Skips lenghty operations (operators, embedding, refining) but retains other functions and printouts. Useful for debugging and checking purposes.
- Compacted output information: individual canditates details only printed with DEBUG keyword, average time per structure and estimate of completion given instead every 20 iterations when saving checkpoints

## 0.4.0 (October 21 2023)
### CONCURRENT OPTIMIZATIONS AND EMBEDDINGS!
### Crest metadynamic conformational searches with mtd>
- xtb_opt now reads output geometry from the trajectory file rather than from xtbopt.xyz. This was needed for multiprocessing capability as it is not possible to override the default name for the optimized geometry (xtbopt.xyz) from the input section.
- Replaced exit() calls with sys.exit()
- xtb_opt now creates a new folder and works inside of it, making it easier to transition into a multiprocessing workflow.
- xtb_pre_opt passes bond constraints to xtb_opt through a string now (constrain_string) instead of a file.
- Implemented parallel (multiprocess) optimization for force_field_refining (2 cores/thread, as many threads to use all CPUs) and optimization_refining (4 cores/thread, as many threads to use all CPUs). Embedder.threads currently not in use.
- Added CHARGE keyword (specify charge to be passed to calculator).
- Removed VMD printout for simple runs (write_vmd) but kept anchor debugging (write_anchor_vmd).
- Added metadynamic conformational search through CREST (mtd_csearch>, or mtd>).
- Changed the way xtb_opt deals with distance constraints, now using very stiff springs instead of exact fixing (better when more then one distance constraint is specified).
- Reinstated fitness_refining after every optimization step, which is now based on a cumulative deviation from the imposed pairing distances.

<!-- - ... mep_relax> BETA
- ... IMAGES kw, also implement it for neb>
- ... [OPENBABEL CONTINUED SUPPORT?] -->