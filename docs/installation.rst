.. _installation:

Installation
============

This program is written in pure Python and it is intended to use with
Python version 3.8.10. The use of a dedicated conda virtual environment
is highly enocouraged.

Prerequisites: before downloading this repository, you should have
installed at least one calculator. At the moment, TSCoDe supports these
calculators:

-  XTB (>=6.3) - recommended
-  ORCA (>=4.2)
-  Gaussian (>=9)
-  MOPAC2016

An additional installation of Openbabel is required, as it provides i/o capabilities via cclib.

XTB (recommended for FF/SE)
---------------------------

This is free software. See the `GitHub
repository <https://github.com/grimme-lab/xtb>`__ and the
`documentation <https://xtb-docs.readthedocs.io/en/latest/contents.html>`__
for how to install it on your machine. The package and its python bindings are available through conda as well.

::

    conda install -c conda-forge xtb xtb-python

It is recommended to optimize the number of
threads used on you system for maximal computaional efficiency (see xtb readthedocs).


ORCA (optional, recommended for DFT)
------------------------------------

This software is only free for academic use at an academic institution.
Detailed instructions on how to install and set up ORCA can be found in
`the official
website <https://sites.google.com/site/orcainputlibrary/setting-up-orca>`__.
Make sure to install and set up OpenMPI along with ORCA if you wish to
exploit multiple cores on your machine **(Note: semiempirical methods
cannot be parallelized in ORCA!)**

Gaussian (optional)
-------------------

This is commercial software available at `the official
website <https://gaussian.com/>`__.


MOPAC2016 (optional, deprecated)
--------------------------------

This software is closed-source but free for academic use. If you qualify
for this usage, you should `request a licence for
MOPAC2016 <http://openmopac.net/form.php>`__. After installation, be
sure to add the MOPAC folder to your system PATH, to access the program
through command line with the "MOPAC2016.exe" command. To test this, the
command ``MOPAC2016.exe`` should return
`this <https://gist.github.com/ntampellini/82224abb9db1c1880e91ad7e0682e34d>`__
message.

Openbabel
---------

This is free software and it is available through conda.

::

    conda install -c conda-forge openbabel

Alternatively, you can download it from `the official
website <http://openbabel.org/wiki/Category:Installation>`__. If you
install the software from the website, make sure to install its Python bindings as well.
You can manually compile these by following the `website
guidelines <https://openbabel.org/docs/dev/Installation/install.html#compile-bindings>`__.

TSCoDe
------

This package is now distributed through PyPI and can be installed through pip.

::

    pip install tscode

After installation, run the guided utility to set up calculation settings (suggested) or manually modify the
`settings.py <https://github.com/ntampellini/TSCoDe/blob/master/tscode/settings.py>`__ file.

::

    python -m tscode --setup

Defaults:

-  FF optimization is turned ON.
-  FF Calculator is XTB

.. To test the installation, you can run the command:

.. ::

..     python -m tscode --test

.. This should take less than 10 minutes on a common computer and point out
.. if any part of the installation is faulted.
