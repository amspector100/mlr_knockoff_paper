import os
import sys

# Add most up-to-date versions of recent packages
file_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.split(file_directory)[0]
kdev_path = None
directory = parent_directory
for _ in range(5):
    if os.path.exists(directory + '/knockpy-dev/'):
        kdev_path = directory + '/knockpy-dev/knockpy/'
        break
    directory = os.path.split(directory)[0]

if kdev_path is None:
    print("knockpy-dev not found, using knockpy from system path")
else:
    print(f"Using knockpy-dev from {kdev_path}")
    sys.path.insert(0, kdev_path)

import knockpy
# This is needed in general
sys.path.insert(0, os.path.abspath(parent_directory))
print(parent_directory)
import mlr_src


print(f"Using knockpy version {knockpy.__version__}.")