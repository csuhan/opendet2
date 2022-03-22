#!/usr/bin/env python
from setuptools import setup

setup(
    name="opendet2",
    version=0.1,
    author="csuhan",
    url="https://github.com/csuhan/opendet2",
    description="Codebase for open set object detection",
    python_requires=">=3.6",
    install_requires=[
        'timm', 'opencv-python'
    ],
)
