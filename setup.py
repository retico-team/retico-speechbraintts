#!/usr/bin/env python3

"""
Setup script.

Use this script to install the core of the retico simulation framework. Usage:
    $ python3 setup.py install

Author: Thilo Michael (uhlomuhlo@gmail.com)
"""

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

exec(open("retico_speechbraintts/version.py").read())

import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

config = {
    "description": "Local speechbrain speech synthesis for retico.",
    "long_description": long_description,
    "long_description_content_type": "text/markdown",
    "author": "Thilo Michael",
    "author_email": "uhlomuhlo@gmail.com",
    "url": "https://github.com/retico-team/retico-speechbraintts",
    "download_url": "https://github.com/retico-team/retico-speechbraintts",
    "version": __version__,
    "python_requires": ">=3.6, <4",
    "keywords": "retico, framework, incremental, dialogue, dialog",
    "install_requires": ["retico_core~=0.2", "speechbrain~=0.5", "numpy~=1.23"],
    "packages": find_packages(),
    "name": "retico-speechbraintts",
    "classifiers": [
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
}

setup(**config)
