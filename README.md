# TSCoDe - Transition State Conformational Docker

<div align="center">

[![License: GNU GPL v3](https://img.shields.io/github/license/ntampellini/TSCoDe)](https://opensource.org/licenses/GPL-3.0)
[![CodeFactor Grade](https://img.shields.io/codefactor/grade/github/ntampellini/TSCoDe)](https://www.codefactor.io/repository/github/ntampellini/tscode)
![Python Version](https://img.shields.io/badge/Python-3.8.10-blue)
![Size](https://img.shields.io/github/languages/code-size/ntampellini/TSCoDe)

![Lines](https://img.shields.io/tokei/lines/github/ntampellini/tscode)
[![PyPI](https://img.shields.io/pypi/v/tscode)](https://pypi.org/project/tscode/)
[![Wheel](https://img.shields.io/pypi/wheel/tscode)](https://pypi.org/project/tscode/)

</div>

<p align="center">

  <img src="docs/images/logo.jpg" alt="TSCoDe logo" class="center" width="500"/>

</p>

TSCoDe is the first systematical conformational embedder for bimolecular and trimolecular chemical reactions. It is able to generate a comprehensive set of both regioisomeric and stereoisomeric poses for molecular arrangements, provided the atoms that will be reacting. It supports both open and cyclical transition states. By feeding the program conformational ensembles, it also generates all conformations combinations. It is thought as a tool to explore TS conformational space in a fast and systematical way, and yield a series of starting points for higher-level calculations.

## :toolbox: Required packages and tools
TSCoDe is written in pure Python. It leverages the numpy library to do the linear algebra required to translate and rotate molecules, the OpenBabel software for performing force field optimization (optional) and the [ASE](https://github.com/rosswhitfield/ase) environment to perform a set of structure manipulations through one of the supported calculators:
-  MOPAC2016
-  ORCA (>=4.2)
-  Gaussian (>=9)
-  XTB (>=6.3)

## Documentation
Documentation on how to install and use the program can be found on [readthedocs](https://tscode.readthedocs.io/en/latest/).


