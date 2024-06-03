#!/usr/bin/env python3
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, "requirements.txt")) as f:
    requirements = f.read().split()

setup(
    name="controversial-stimuli",
    version="0.0.0",
    description="A framework used to generate most discriminative stimuli",
    author="Max Burg, Thomas Zenkel",
    author_email="max.burg@bethgelab.org",
    license="MIT",
    url="https://github.com/ecker-lab/most-discriminative-stimuli",
    packages=find_packages(exclude=[]),
    install_requires=requirements,
)
