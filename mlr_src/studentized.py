import numpy as np
import knockpy
from knockpy.knockoff_stats import FeatureStatistic, LassoStatistic, combine_Z_stats, default_regularization
import warnings
import sklearn.linear_model

class ElasticNetStatistic(FeatureStatistic):

    def __init__(self, mx: bool, **kwargs):
        super().__init__(**kwargs)
        self.mx = mx

    def fit(self, X, Xk, y, groups, **kwargs):
        # initialize feature statistics
        n, p = X.shape
        if groups is None:
            groups = np.arange(1, p+1, 1)
        
        # Step 0: concatenate and get features
        features = np.concatenate([X, Xk], axis=1)
        inds, rev_inds = knockpy.utilities.random_permutation_inds(2 * p)
        features = features[:, inds]


        # Step 1: fit elasticnet
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            if self.mx:
                self.model = sklearn.linear_model.ElasticNetCV(
                    alphas=10,
                    cv=5,
                )
            else:
                # can't use CV, use heuristic from https://arxiv.org/pdf/1508.02757
                alpha = default_regularization(X, Xk, y)
                self.model = sklearn.linear_model.ElasticNet(alpha=alpha)
                
            self.model.fit(features, y)

        # Step 2: retrieve coefs
        Z = self.model.coef_
        self.Z = Z[rev_inds]
        self.W = combine_Z_stats(
            self.Z,
            groups=groups,
            antisym=kwargs.get('antisym', 'cd'),
            group_agg=kwargs.get('group_agg', 'sum'),
        )
        return self.W

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
    kfe = knockpy.knockoff_filter.KnockoffFilter(
        fstat=ElasticNetStatistic(),
        ksampler='gaussian',
    )
    kfe.forward(
        X=dgp.X,
        y=dgp.y,
        Sigma=dgp.Sigma,
    )
    print("ElasticNet:")
    print(kfe.fstat.W)
    print("rejections:", kfe.make_selections(kfe.fstat.W, 0.1).sum())
    print("Success!")