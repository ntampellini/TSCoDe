## TSCoDe Changelog

### 0.0.1 (10 Aug 2021)
- First release

### 0.0.2 (10 Aug 2021)
- If pivots decrease during a bend, an exception is raised. Future versions might have a different behavior in this scenario.

### 0.0.3 (10 Aug 2021)
- setup.py bugfix.

### 0.0.4 (Aug 2021)
- SADDLE keyword implementation.
- Added keywords print at top of log
- Pairings are now of two types: reactive atoms (a, b, c) or NCIs (x, y, z). The latter are adjusted when specifying distances with DIST but are left free to reach their equilibrium distance (HalfSpring constraint + additional relax).
- Major code cleaning, refactoring and reordering
- Added solvent support for calculators (SOLVENT keyword)
- Dihedral embeds now support both the SADDLE and NEB keywords
- Similar structures are now pruned in a rational way: the best looking is kept (fast_score)