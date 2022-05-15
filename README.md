# TSCoDe - Transition State Conformational Docker

<div align="center">

[![License: GNU GPL v3](https://img.shields.io/github/license/ntampellini/TSCoDe)](https://opensource.org/licenses/GPL-3.0)
[![CodeFactor Grade](https://img.shields.io/codefactor/grade/github/ntampellini/TSCoDe)](https://www.codefactor.io/repository/github/ntampellini/tscode)
![Python Version](https://img.shields.io/badge/Python-3.8.10-blue)
![Size](https://img.shields.io/github/languages/code-size/ntampellini/TSCoDe)

![Lines](https://img.shields.io/tokei/lines/github/ntampellini/tscode)
[![PyPI](https://img.shields.io/pypi/v/tscode)](https://pypi.org/project/tscode/)
[![Wheel](https://img.shields.io/pypi/wheel/tscode)](https://pypi.org/project/tscode/)
[![Documentation Status](https://readthedocs.org/projects/tscode/badge/?version=latest)](https://tscode.readthedocs.io/en/latest/?badge=latest)

</div>

<p align="center">

  <img src="docs/images/logo.jpg" alt="TSCoDe logo" class="center" width="500"/>

</p>

TSCoDe is a systematical conformational embedder for small molecules. It helps computational chemists build transition states and binding poses precisely in an automated way. It is thought as a tool to explore complex multimolecular conformational space fast and systematically, and yield a series of starting points for higher-level calculations.

Since its inclusion of many subroutines and functionality, it also serves as a computational toolbox
to automate various routine tasks, via either MM, semiempirical or DFT methods.

## :toolbox: Dependencies
TSCoDe is written in pure Python. It leverages the Numpy and Numba libraries to perform the linear algebra required to translate and rotate molecules and the [ASE](https://github.com/rosswhitfield/ase) environment to perform a set of structure manipulations. It supports various external calculators (at least one required):

-  MOPAC2016
-  ORCA (>=4.2)
-  Gaussian (>=9)
-  XTB (>=6.3)
-  Openbabel (3.1.0)

## Documentation
Documentation on how to install and use the program can be found on [readthedocs](https://tscode.readthedocs.io/en/latest/index.html).