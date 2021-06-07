# TSCoDe - Transition State Conformational Docker

<div align="center">

 [![License: GNU GPL v3](https://img.shields.io/github/license/ntampellini/TSCoDe)](https://opensource.org/licenses/GPL-3.0) [![CodeFactor Grade](https://img.shields.io/codefactor/grade/github/ntampellini/TSCoDe)](https://www.codefactor.io/repository/github/ntampellini/tscode)

</div>

<p align="center">

  <img src="images/logo.jpg" alt="TSCoDe logo" class="center" width="500"/>

</p>

TSCoDe is the first systematical conformational embedder for bimolecular and trimolecular chemical reactions. It is able to generate a comprehensive set of both regioisomeric and stereoisomeric poses for molecular arrangements, provided the atoms that will be reacting. It supports both open and cyclical transition states. By feeding the program conformational ensembles, it also generates all conformations combinations. It is thought as a tool to explore TS conformational space in a fast and systematical way, and yield a series of starting points for higher-level calculations.

**NOTE: structures obtained from TSCoDe are not proper transition states (most of the times) but are often quite close. The program is intended to yield and rank poses, not TSs. In this way, the computational chemist can skip the error-prone phase of molecular embedding and proceed to the most appropriate higher-level calculation step.**

### Required packages and tools
TSCoDe is written in pure Python, ~~with some libraries optionally boosted via Cython~~. It leverages the numpy library to do the linear algebra required to translate and rotate molecules, the OpenBabel software for performing force field optimization and the [ASE](https://github.com/rosswhitfield/ase) environment to perform a set of structure manipulations through the semiempirical [MOPAC2016](http://openmopac.net/MOPAC2016.html) calculator. While the former is free software, the latter is only free for academic use, and a license must be requested via the MOPAC website (see *Installation*).

## :green_circle: What the program can do (well)
**Generate accurately spaced poses** for bimolecular and trimolecular transition states of organic molecules, also considering structural deformation. If a transition state is already in hand, the distance between reactive atoms can be specified, so as to obtain all the stereo/regioisomeric analogs with precise molecular spacings. TSCoDe is best suited for modelizations that involve many transition state regio- and stereoisomers, where the combination of reagents conformations is an important aspect in transition state building. The program yields many atomic arrangements that can be evaluated by the computational chemist, aiding them in exploring all conformational and regio/stereochemical reaction space.

## :yellow_circle: What the program can do (sometimes)
**Infer differential NCIs** - After the poses generation, the program can be told to infer the non-covalent interactions (NCIs) between all generated structures (`NCI` keyword). If a particular NCI is not shared by all structures, that is reported. If a particularly strong NCI is present only in a few TSs, this function can be a handy tool for tracing the source of selectivity in a given chemical reaction.

**Generate proper transition state structures** (semiempirical level) - After poses generation, these structures can be used to try to directly obtain transition state structures at the semiempirical level chosen. This is not a default behavior, and it is invoked by the `NEB` keyword. A climbing image nudged elastic band (CI-NEB) transition state search is performed after inferring both reagents and products for each individual pose. This entire process is of course challenging to automate completely, and can be prone to failures. Associative reactions, where two (or three) molecules are bound together (or strongly interacting) after the TS, with no additional species involved, tend to give good results. For example, cycloaddition reactions are great candidates while atom transfer reactions (*i.e.* epoxidations) are not.

## :red_circle: What the program cannot do
**Perfectly replicate TS structures at high levels of theory** - As the program exploits MOPAC to perform calculations, final geometries arise from constrained optimizations at a semiempirical level of theory (default is PM7). They are therefore not granted to perfectly replicate higher-level calculations. However, constrained optimizations through external programs are meant to be used to refine these structures and obtain TSs.


## Installation (Windows, Linux and Mac)

This program is written in pure Python and it is intended to use with Python version 3.8.10. The use of a dedicated conda virtual environment is highly suggested.

Prerequisites: before downloading this repository, you should have installed both Openbabel (required for Force Field minimizations) and MOPAC2016 (required for semiempirical calculations).

### Openbabel

This is free software you can download from [the official website](http://openbabel.org/wiki/Category:Installation). After you have installed the software, make sure to install its Python bindings. You can manually compile these by following the [website guidelines](https://openbabel.org/docs/dev/Installation/install.html#compile-bindings), but *by far* the easiest procedure is just using conda:

    conda install -c conda-forge openbabel

### MOPAC2016

This software is only free for academic use. If you qualify for this usage, you should [request a licence for MOPAC2016](http://openmopac.net/form.php). After installation, be sure to add the MOPAC folder to your system PATH, to access the program through command line with the "MOPAC2016.exe" command. To test this, the command `MOPAC2016.exe` should return [this](https://gist.github.com/ntampellini/82224abb9db1c1880e91ad7e0682e34d) message.

### TSCoDe

I you have Git installed, you can directly clone the repository: *(otherwise download the code and unpack it)*

    git clone https://github.com/ntampellini/TSCoDe
    
Open a command shell, move to the ./TSCoDe folder and install the requirements.

    cd TSCoDe

    pip install -r requirements.txt

To test the installation, you can run the provided test in the test folder:

    python tests/run_tests.py

This should take less than 10 minutes on a modern laptop and point out if any part of the installation is faulted.

## Usage

The program can be run from terminal, both from the TSCoDe folder

    python tscode.py <path_to_input_folder>/myinput.txt [custom_title]

or from the input folder.

    python <path_to_TSCoDe_folder>/tscode.py myinput.txt [custom_title]

A custom title for the run can be optionally provided, otherwise a time stamp will be used.

### Example of `myinput.txt`

    DIST(a=2.135,b=1.548,c=1.901) NEB
    maleimide.xyz 0a 5b
    mopac>HCOOH.xyz 4b 1c
    csearch>dienamine.xyz 6a 23c

    # Free number of comment lines!

    # First pairing (a) is the C-C reactive distance
    # Second and third pairings (b, c) are the
    # hydrogen bonds bridging the two partners.

    # Structure of HCOOH.xyz will be optimized with MOPAC before running TSCoDe
    # A conformational analysis will be performed on dienamine.xyz before running TSCoDe
    
<p align="center">

<img src="images/tri.PNG" alt="Example input file" class="center" width="500"/>

</p>

### Input formatting
The program input can be any text file.
- Any blank line will be ignored
- Any line starting with `#` will be ignored
- Keywords, if present, need to be on first non-blank, non-comment line
- Then, two or three molecule files are specified, along with their reactive atoms indexes, and eventually their pairings

TSCoDe can work with all molecular formats read by [cclib](https://github.com/cclib/cclib), but best practice is using only the `.xyz` file format, particularly for multimolecular files containing different conformers of the same molecule. **The reactive indexes specified are counted starting from zero!** If the molecules are specified without reactive indexes, a pop-up ASE GUI window will guide the user into manually specifying the reactive atoms after running the program.
 
Reactive atoms supported are `C, H, O, N, P, S, F, Cl, Br, I`. Reactions can be of four kinds:
- **Two molecules, one reactive atom each** - "string embed" (*i.e.* SN2 reactions)
- **Two molecules, one with a single reactive atom and the other with two reactive atoms** - "chelotropic embed" (*i.e.* epoxidations)
- **Two molecules, two reactive atoms each** - "cyclical embed" (*i.e.* Diels-Alder reactions)
- **Three molecules, two reactive atoms each** - "cyclical embed" (*i.e.* reactions where two partners are bridged by a carboxylic acid like the example above)

After each reactive index, it is possible to specify a letter (`a`, `b` or `c`) to represent the "flag" of that atom. If provided, the program will only yield the regioisomers that respect those atom pairings. For "chelotropic embeds", one could specify that a single atom has two flags, for example the oxygen atom of a peracid, like `4ab`.

If a `NEB` calculation is to be performed on a trimolecular transition state, the reactive distance "scanned" is the first imposed (`a`). See `NEB` keyword in the keyword section.

Molecule files can be preceded by *operators*, like `csearch>molecule.xyz`. They operate on the input file before it is fed to TSCoDe. It is important not to include any space character between the operator and the molecule name.

- **`mopac>`** - Performs a MOPAC optimization of the structure(s) before using it in TSCoDe. Generates a new `molecule_opt.xyz` file with the optimized coordinates.

- **`csearch>`** - Performs a simple Openbabel conformational search and optimizes all conformers with MOPAC. Then, a maximum of five best conformers are used to run TSCoDe. Generates a new `molecule_confs.xyz` file with all optimized conformers.
  
### Good practice and suggested options

When modeling a reaction through TSCoDe, I suggest following this guidelines:

- Assess that the reaction is supported by TSCoDe. See Input formatting: monomolecular reactions are not yet supported.

- Obtain molecular structures in .xyz format. If more conformers are to be used, they must be in a multimolecular .xyz file, and atom ordering must be consistent throughout all structures. I suggest using a maximum of five conformers for each structure (ideally less) as these calculations have a steep combinatorial scaling.

- If, in a trimolecular TS, two molecules are interacting but the TS is not cyclical (*i.e.* thiourea-like organocatalytic carbonyl activations) then TSCoDe input needs to be a bimolecular TS, where the first molecule is the interaction complex (*i.e.* ketone + thiourea) and the second one is the nucleophile.

- Understand what atoms are reacting for each structure and record their index (**starting from 0!**). If you are unsure of reactive atomic indexes, you can run a test input without indexes, and the program will ask you to manually specify them from the ASE GUI by clicking. This is not possible if you are running TSCoDe from a command line interface (CLI). When choosing this option, it is not possible to specify atom pairings. Therefore, I suggest using this option only to check the reactive atoms indexes and then building a standard input file.

- Optionally, after specifying reactive indexes, the `CHECK` keyword can be used. A series of pop-up ASE GUI windows will be displayed, showing each molecule with a series of red dots around the reactive atoms chosen. This can be used to check "orbital" positions or conformer reading faults (scroll through conformers with page-up and down buttons). Program will terminate after the last visualization is closed.

- By default, TSCoDe parameters are optimized to yield good results without specifying any keyword nor atom pairing. However, I strongly encourage to specify all the desired pairings and use the `DIST` keyword in order to speed up the calculation and achieve better results, respectively. For example, the trimolecular reaction in the example above is described with all three atomic pairings (`a`, `b` and `c`) and their distances. These can come from a previous higher-level calculation or can be inferred by similar reactions. If they are not provided, a guess is performed by reading the `parameters.py` file.

- If the reaction involves big molecules, or if there are a lot of conformations, a preliminar calculation using the `NOOPT` keyword may be a good idea to see how many structures are generated and would require MOPAC optimization in a standard run.

- If TSCoDe does not find enough suitable candidates (or any at all) for the given reacion, most of the times this is because of compenetration pruning. This mean that a lot of structures are generated, but all of them have some atoms compenetrating one into the other, and are therefore discarded. A solution could be to loosen the compenetration rejection citeria (`CLASHES` keyword, not recommended) or to use the `SHRINK` keyword (recommended, see keywords section). Note that `SHRINK` calculations will be loger, as MOPAC/ASE distance-refining optimizations will require more iterations to reach target distances.

### Keywords

Keywords are divided by at least one blank space. Some of them are self-sufficient (*i.e.* `NCI`), while some others require an additional input (*i.e.* `STEPS=10` or `DIST(a=1.8,b=2,c=1.34)`). In the latter case, whitespaces inside the parenthesis are NOT allowed. Floating point numbers are to be expressed with points like `3.14`, while commas are used to divide keyword arguments where more than one are accepted, like in `DIST`.

- **`BYPASS`** - Debug keyword. Used to skip all pruning steps and directly output all the embedded geometries.

- **`CHECK`** - Visualize the input molecules through the ASE GUI, to check orbital positions or reading faults.

- **`CLASHES`** - Manually specify the max number of clashes and/or the distance threshold at which two atoms are considered clashing. The more forgiving, the more structures will reach the geometry optimization step. Syntax: `CLASHES(num=3,dist=1.2)`
  
- **`DEEP`** - Performs a deeper search, retaining more starting points for calculations and smaller turning angles. Equivalent to `THRESH=0.3 STEPS=24 CLASHES=(num=3,dist=1.2)`. Use with care!

- **`DIST`** - Manually imposed distance between specified atom pairs, in Angstroms. Syntax uses parenthesis and commas: `DIST(a=2.345,b=3.67,c=2.1)`

- **`KCAL`** - Trim output structures to a given value of relative energy. Syntax: `KCAL=n`, where n can be an integer or float.

- **`LET`** - Overrides safety checks that prevent the program from running too large calculations.

- **`LEVEL`** - Manually set the MOPAC theory level to be used, default is PM7. Syntax: `LEVEL=PM7`

- **`MMFF`** - Use the Merck Molecular Force Field during the OpenBabel pre-optimization (default is UFF).

- **`NCI`** - Estimate and print non-covalent interactions present in the generated poses.

- **`NEB`** - Perform an automatical climbing image nudged elastic band (CI-NEB) TS search after the partial optimization step, inferring reagents and products for each generated TS pose. These are guessed by approaching the reactive atoms until they are at the right distance, and then partially constrained (reagents) or free (products) optimizations are carried out to get the start and end points for a CI-NEB TS search. For trimolecular transition states, only the first imposed pairing (a) is approached - *i.e.* the C-C reactive distance in the example above. This `NEB` option is only really usable for those reactions in which two (or three) molecules are bound together (or strongly interacting) after the TS, with no additional species involved. For example, cycloaddition reactions are great candidates while atom transfer reactions (*i.e.* epoxidations) are not. Of course this implementation is not always reliable, and it is provided more as an experimenting tool than a definitive feature.

- **`NEWBONDS`** - Manually specify the maximum number of "new bonds" that a TS structure can have to be retained and not to be considered scrambled. Default is 1. Syntax: `NEWBONDS=1`

- **`NOOPT`** - Skip the optimization steps, directly writing structures to file.

- **`ONLYREFINED`** - Discard structures that do not successfully refine bonding distances.

- **`RIGID`** - Does not apply to "string" embeds. Avoid bending structures to better build TSs.

- **`ROTRANGE`** - Does not apply to "string" embeds. Manually specify the rotation range to be explored around the structure pivot. Default is 90. Syntax: `ROTRANGE=90`

- **`SHRINK`** - Exaggerate orbital dimensions during embed, scaling them by a specified factor. If used as a single keyword (`SHRINK`), orbital dimensions are scaled by a factor of one and a half. A syntax like `SHRINK=3.14` allows for custom scaling. This scaling makes it easier to perform the embed without having molecules clashing one into the other. Then, the correct distance between reactive atom pairs is achieved as for standard runs by spring constraints during MOPAC optimization. The larger the scaling, the more the program is likely to find at least some transition state poses, but the more time-consuming the step of distance refinement.

- **`STEPS`** -  Manually specify the number of steps to be taken in scanning rotations. For string embeds, the range to be explored is the full 360°, and the default `STEPS=24` will perform 15° turns. For cyclical and chelotropic embeds, the rotation range to be explored is +-`ROTRANGE` degrees. Therefore the default values, equivalent to `ROTRANGE=90 STEPS=9`, will perform nine 20 degrees turns.

- **`SUPRAFAC`** - Only retain suprafacial orbital configurations in cyclical TSs. Thought for Diels-Alder and other cycloaddition reactions.

- **`THRESH`** - RMSD threshold (Angstroms) for structure pruning. The smaller, the more retained structures. Default is 0.5 A. Syntax: `THRESH=n`, where n is a number.
