# MLR Paper

This repository contains all the code to replicate the experiments and real data analyses from Spector and Fithian (2022). Running these scripts requires a python environment with ``knockpy`` installed (version 1.3+): see https://github.com/amspector100/knockpy for installation.

## Overview

The directory ``mlr_src`` contains extraneous functions used in the simulations. However, the core contribution of the paper (MLR statistics) is implemented and published in the ``knockpy`` package to ease installation. 

The directory ``sims`` contains the code which actually runs the simulations. It also contains a ``all_sims.sh`` files which will replicate the exact simulation settings in the paper. The exception is that the data for a few plots was simulated directly in the ``final-plots/final_plots.ipynb`` notebook.

The code needed to replicate the three real data applications are in the ``real_data/`` subdirectory. 

In general, all ``.py`` and ``.sh`` files should be called from the directory in which they reside. For example, one should navigate to the ``real_data/`` subdirectory before running ``python3.9 hiv_data.py`` to replicate the HIV data analysis.

## Figure by figure

1. Figure 1: see ``final-plots/final_plots.ipynb``.
2. Figure 2: generated using ``sims/all_sims.sh`` using ``LINEAR_FX_ARGS`` and ``LINEAR_MX_ARGS``.
3. Figure 3: generated using ``sims/all_sims.sh`` using ``VP_ARGS``.
4. Figure 4: generated using ``sims/all_sims.sh`` using ``SPARSE_ARGS``.
5. Figure 5: see ``final-plots/final_plots.ipynb``.
6. Figure 6: generated using ``sims/all_sims.sh`` using ``NONLIN_ARGS``.
7. Figure 7: generated using ``sims/all_sims.sh`` using ``LOGISTIC_ARGS``.
8. Figure 8: generated from ``real_data/hiv_data.py``.
9. Figure 9: generated from ``real_data/fund_rep.py``.
10. Figure 10: generated from ``real_data/nodewise_knock.py``.
11. Figure 11: generated using ``sims/all_sims.sh`` using ``NONLIN_ARGS``.
12. Figures 12-15: generated using ``real_data/hiv_data.py`` and ``final-plots/final_plots.ipynb``. 
