## TSCoDe Changelog

### 0.0.1 (10 Aug 2021)
- First release

### 0.0.2 (10 Aug 2021)
- If pivots decrease during a bend, an exception is raised. Future versions might have a different behavior in this scenario.

### 0.0.3 (10 Aug 2021)
- setup.py bugfixes.

### 0.0.5 (1 Sep 2021)
- SADDLE keyword implementation.
- Added keywords print at top of log
- Pairings are now of two types: reactive atoms (a, b, c) or NCIs (x, y, z). The latter are adjusted when specifying distances with DIST but are left free to reach their equilibrium distance (HalfSpring constraint + additional relax).
- Major code cleaning, refactoring and reordering
- Added solvent support for calculators (SOLVENT keyword)
- Dihedral embeds now support both the SADDLE and NEB keywords
- Similar structures are now pruned in a rational way: the best looking is kept (fast_score)

### 0.0.6 (19 Oct 2021)
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

### 0.0.7 (20 Oct 2021)
- Import bugfix