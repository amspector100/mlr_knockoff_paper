# MLR Paper

This repository contains all the code to replicate the experiments and real data analyses from Spector and Fithian (2022). Running these scripts requires a python environment with ``knockpy`` installed (version 1.3+): see https://github.com/amspector100/knockpy for installation.

## Overview

The directory ``mlr_src`` contains extraneous functions used in the simulations. However, the core contribution of the paper (MLR statistics) is implemented and published in the ``knockpy`` package to ease installation. 

The directory ``sims`` contains the code which actually runs the simulations. It also contains a ``all_sims.sh`` files which will replicate the exact simulation settings in the paper. The exception is that the data for a few plots was simulated directly in the ``final-plots/final_plots.ipynb`` notebook.

The code needed to replicate the three real data applications are in the ``real_data/`` subdirectory. 

In general, all ``.py`` and ``.sh`` files should be called from the directory in which they reside. For example, one should navigate to the ``real_data/`` subdirectory before running ``python3.9 hiv_data.py`` to replicate the HIV data analysis.