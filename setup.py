"""Editable-install compatibility for the system pip bundled with Python 3.9."""

from setuptools import find_packages, setup

setup(
    name="skillrsi-m1",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.9",
    entry_points={"console_scripts": ["skillrsi-m1=skillrsi_m1.cli:main", "skillrsi-m2=skillrsi_m1.m2:main"]},
)
