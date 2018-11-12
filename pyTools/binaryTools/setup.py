from __future__ import absolute_import
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("readwriteC2_220", ["dataCythonIO.pyx"])]
)
