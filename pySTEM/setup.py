#!/usr/bin/env python
import setuptools
from distutils.core import setup, Extension
import numpy.distutils.misc_util
with open("README.md", "r") as fh:
    long_description = fh.read()
__version__ = '0.0.13'

setup(name='pystem',version=__version__,
      ext_modules=[Extension("_stemdescriptor", ["_stemdescriptor.c", "calculate_descriptor.c"],
                   depends=['calculatedescriptor.h'],extra_compile_args=['-fopenmp'])],
      py_modules=['stemsegmentation','stemdescriptor','stemclustering','stempower_spectrum',
                  'stemrotational_symmetry_descriptors','stemreflection_symmetry_descriptors'],
      include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
      description = 'A python module for segmentation of STEM images',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url = 'https://github.com/NingWang1990/pySTEM',
      author = 'Ning Wang',
      author_email = 'nwang@mpie.de',
      maintainer = 'Ning Wang',
      maintainer_email = 'nwang@mpie.de',
      license = 'GNU GPLv3',
      classifiers = [
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Topic :: Security :: Cryptography',
        'Topic :: Software Development :: Libraries :: Python Modules',
      ]
    )

