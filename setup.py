from setuptools import setup,find_packages
import os
setup(
    name='myData',
    version='0.1',
    packages=['Data'],
)

os.system('rm -rf build')
os.system('rm -rf dist')
os.system('rm -rf *.egg-info')