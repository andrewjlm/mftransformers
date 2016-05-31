#!/usr/bin/env python
# -*- coding: utf-8 -*-

from distutils.core import setup

setup(name='mftransformers',
      version='0.1dev',
      description='scikit-learn wrapper around PyMF methods',
      author='Andrew Mullins',
      author_email='mullinsajl@gmail.com',
      url='https://github.com/andrewjlm/mftransformers',
      packages=['mftransformers'],
      install_requires=['PyMF'])
