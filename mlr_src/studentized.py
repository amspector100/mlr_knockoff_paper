import numpy as np
import knockpy
from knockpy.knockoff_stats import LassoStatistic, combine_Z_stats
import warnings

class StudentizedLassoStatistic(LassoStatistic):

    def fit(self, X, Xk, y, nboot=50, **kwargs):
        # initial feature statistics
        W = super().fit(X, Xk, y, **kwargs)
        Z = self.Z.copy()
        self.W_unstudentized = W
        self.Z_unstudentized = Z
        
        # bootstrap
        self.Z_boot = np.zeros((nboot, len(Z)))
        for i in range(nboot):
            # resample with replacement
            inds = np.random.choice(len(y), size=len(y), replace=True)
            # fit model
            super().fit(X[inds], Xk[inds], y[inds], **kwargs)
            Z_boot = self.Z.copy()
            # store
            self.Z_boot[i, :] = Z_boot
        
        # studentize
        ses = np.std(self.Z_boot, axis=0)
        if np.all(ses == 0):
            warnings.warn("All bootstrap standard errors are zero; falling back to unstudentized.")
            ses = np.ones(len(self.Z))
        ses[ses == 0] = ses[ses > 0].min()
        self.Z = Z / ses

        # aggregate and such
        self.W = combine_Z_stats(
            self.Z,
            groups=kwargs.get('groups', None),
            antisym=kwargs.get('antisym', 'cd'),
            group_agg=kwargs.get('group_agg', 'sum'),
        )

        # return
        return self.W
        
# test
if __name__ == "__main__":
    dgp = knockpy.dgp.DGP()
    dgp.sample_data(n=100, p=50, method='ar1', a=5, b=1, sparsity=0.5)
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
    print("rejections:", kfilter.make_selections(kfilter.fstat.W_unstudentized, 0.1).sum())
    print("Studentized:")
    print(kfilter.fstat.W)
    print("rejections:", kfilter.make_selections(kfilter.fstat.W, 0.1).sum())
    print("Success!")