import setuptools, os
from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)
setup(ext_modules = cythonize('compenetration.pyx'),
      include_dirs=[np.get_include()])