import os
from setuptools import setup, find_packages


with open(os.path.join("README.md"), "r") as f:
    readme = f.read()

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name="beebo",
    version="0.0.1",
    description="An acquisition function for Bayesian optimization",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Anonymous",
    packages=find_packages(),
    python_requires=">=3.6, <4",
    install_requires=requirements,
)