from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext as _build_ext
import subprocess
import os
from pathlib import PurePath

current_directory = os.path.dirname(os.path.abspath(__file__))


def build_extensions():
    """Used the subprocess module to compile/install the C software."""
    src_path = PurePath(current_directory)/'interpolation2_3D'

    subprocess.run('./compile.bash', cwd=src_path, shell=False, check=True,
            stdout = subprocess.PIPE, stderr=subprocess.STDOUT)



class build_ext(_build_ext):
    """Custom handler for the 'build' command."""
    def run(self):
        build_extensions()
        super().run()

# Optional project description in README.md:
try:

    with open(os.path.join(current_directory, 'README.md'), encoding='utf-8') as f:

        long_description = f.read()

except Exception:

    long_description = 'EAGLE-like post-process code'

setup(

# Project name:
name='interpolation2-3D',

# Packages to include in the distribution:
packages=find_packages(','),
# packages=['egl'],

# Project version number:
version='13.3.3',

# List a license for the project, eg. MIT License
license='',

# Short description of your library:
description='2D fast interpolation of 3D data, compatible with NUMBA',

# Long description of your library:

long_description=long_description,

long_description_content_type='text/markdown',

# Your name:
author='Andrea Negri',

# Your email address:
author_email='anegri@iac.es',

# Link to your github repository or website:
# url='',

# Download Link from where the project can be downloaded from:
# download_url='',

# List of keywords:
# keywords=[]

# List project dependencies:
install_requires=['numpy','scipy', 'pathlib', 'numba', 'cffi'],


cmdclass={'build_ext': build_ext}
)

