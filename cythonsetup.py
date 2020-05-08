# distutils: language = c++

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

#extensions = [Extension('Evolutionary Algorithm', ["mutator.pyx"])]

setup(
    name='Evolutionary Algorithm',
    description = 'A code to solve 814-2 using Cython-module',
    ext_modules=cythonize('mutator.pyx', annotate = True, language_level=3),
    include_dirs=[numpy.get_include()]
)

# python cythonsetup.py build_ext --inplace
'''
Add the following compile options at the beginning of .pyx file:
# cython: profile=True
# cython: linetrace=True
# cython: binding=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
'''