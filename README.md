# MLR Paper

This repository contains all the code to replicate the experiments and real data analyses from Spector and Fithian (2022). Running these scripts requires a python environment with ``knockpy`` installed (version 1.3+): see https://github.com/amspector100/knockpy for installation.

## Overview

The directory ``mlr_src`` contains extraneous functions used in the simulations. However, the core contribution of the paper (MLR statistics) is implemented and published in the ``knockpy`` package to ease installation. 

The directory ``sims`` contains the code which actually runs the simulations. It also contains a ``all_sims.sh`` files which will replicate the exact simulation settings in the paper. The exception is that the data for a few plots was simulated directly in the ``final-plots/final_plots.ipynb`` notebook.

