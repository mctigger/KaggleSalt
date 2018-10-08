# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

try:
    long_description = open("README.md").read()
except IOError:
    long_description = ""

setup(
    name="ela",
    version="0.1.0",
    description="A lightweight data-augmentation library for machine learning",
    license="MIT",
    author="Tim Joseph",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-image',
        'scipy'
    ],
    long_description=long_description,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
    ]
)
