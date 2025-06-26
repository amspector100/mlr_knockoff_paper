import numpy as np
import pandas as pd
from tqdm import tqdm
import knockpy
from knockpy.knockoff_stats import LassoStatistic
import warnings

class StudentizedLassoStatistic(LassoStatistic):

    def fit(self, X, Xk, y, nboot=50, **kwargs):
        # initial feature statistics
        W = super().fit(X, Xk, y, **kwargs)
        
        # bootstrap
        self.W_boot = np.zeros((nboot, len(W)))
        for i in range(nboot):
            # resample with replacement
            inds = np.random.choice(len(y), size=len(y), replace=True)
            # fit model
            W_boot = super().fit(X[inds], Xk[inds], y[inds], **kwargs)
            # store
            self.W_boot[i, :] = W_boot
        
        # studentize
        ses = np.std(self.W_boot, axis=0)
        if np.all(ses == 0):
            warnings.warn("All bootstrap standard errors are zero; falling back to unstudentized.")
            ses = np.ones(len(W))
        ses[ses == 0] = ses[ses > 0].min()
        self.W_unstudentized = W
        self.W = W / ses

        # return
        return self.W
        
# test
if __name__ == "__main__":
    dgp = knockpy.dgp.DGP()
    dgp.sample_data(n=100, p=50)
    kfilter = knockpy.knockoff_filter.KnockoffFilter(
        fstat=StudentizedLassoStatistic(),
        ksampler='gaussian',
    )
    kfilter.forward(
        X=dgp.X,
        y=dgp.y,
        Sigma=dgp.Sigma,
    )
    print("Unstudentized:")
    print(kfilter.fstat.W_unstudentized)
    print("Studentized:")
    print(kfilter.W)
    print("Success!")