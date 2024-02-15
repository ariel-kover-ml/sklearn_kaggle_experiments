from setuptools import find_packages, setup

setup(
    name="sklearn_ext",
    version="0.1.0",
    description="Sklearn Extension",
    author="Ariel Kover",
    license="MIT",
    package_dir={"": "src"}, # Empty string means root directory
    packages=find_packages(where="src")
)
