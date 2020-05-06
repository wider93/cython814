# distutils: language = c++

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

#extensions = [Extension('Evolutionary Algorithm', ["mutator.pyx"])]

setup(
    name='Evolutionary Algorithm',
    description = 'A code to solve 814-2 using Cython-module',
    ext_modules=cythonize('mutator_partof.pyx', annotate = True, language_level=3),
    include_dirs=[numpy.get_include()]
)

# python cythonsetup.py build_ext --inplace