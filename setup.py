from setuptools import setup, find_packages

setup(
name='Machine Learning Algorithm Develop Python',
version='1.0',
author='Tony Silva',
author_email='tonysilva.ou@gmail.com',
packages=find_packages(), #exclude=()
setup_requires=['pytest-runner'],
tests_require=['pytest'],
)