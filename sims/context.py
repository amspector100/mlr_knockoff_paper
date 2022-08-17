import os
import sys

# Add mlr_src to path
file_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.split(file_directory)[0]
grandparent_dir = os.path.split(parent_directory)[0]
# No need for this if knockpy is installed via pip
sys.path.insert(0, os.path.abspath(grandparent_dir + 'knockpy-dev/knockpy/'))
import knockpy
# This is needed in general
sys.path.insert(0, os.path.abspath(parent_directory))
import mlr_src


print(f"Using knockpy version {knockpy.__version__}")