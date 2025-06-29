import os
import sys

# Add knockpy-dev to path
# No need for this if knockpy is installed via pip
file_directory = os.path.dirname(os.path.abspath(__file__))
directory = file_directory
kdev_path = None
for i in range(5):
    directory = os.path.split(directory)[0]
    if os.path.exists(directory + '/knockpy-dev/'):
        kdev_path = directory + '/knockpy-dev/knockpy/'
        sys.path.insert(0, kdev_path)
        break
if kdev_path is None:
    print("knockpy-dev not found, using knockpy from system path")
else:
    print(f"Using knockpy-dev from {kdev_path}")
    sys.path.insert(0, kdev_path)

import knockpy
# This is needed in general
parent_directory = os.path.split(file_directory)[0]
sys.path.insert(0, os.path.abspath(parent_directory))
import mlr_src


print(f"Using knockpy version {knockpy.__version__}.")