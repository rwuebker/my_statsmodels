import os
import sys
import numpy as np
import statsmodels.api as sm
import statsmodels.tools as sm_tools

p = os.getcwd()
if p not in sys.path:
    sys.path.insert(0, p)

from models import OLS
from models import calc_IC_metrics
from models import calc_log_likelihood


def test_regular_ols():
    rng = np.random.default_rng(12345)
    randoms = rng.normal(0, 20, 100).reshape(-1, 1)
    x = np.linspace(0, 100, 100).reshape(-1, 1)
    y = 3*x + randoms

    ols = OLS(x, y)
    ols.fit()

    sm_X = sm.add_constant(x)
    sm_results = sm.OLS(y, sm_X).fit()
    assert np.allclose(ols.beta, sm_results.params)
    assert np.allclose(ols.std_errors, sm_results.bse)

    X = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
    y_pred = ols.predict(X)
    assert np.allclose(ols.predict(X), sm_results.predict())


def test_robust_ols():
    # uses the white errors
    rng = np.random.default_rng(12345)
    randoms = rng.normal(0, 20, 100).reshape(-1, 1)
    x = np.linspace(0, 100, 100).reshape(-1, 1)
    y = 3*x + randoms

    ols = OLS(x, y)
    ols.fit(cov_type='white')

    sm_X = sm.add_constant(x)
    sm_results = sm.OLS(y, sm_X).fit()
    assert np.allclose(ols.beta, sm_results.params)
    assert np.allclose(ols.std_errors, sm_results.HC1_se)


def test_calc_IC_metrics():
    nobs = 100
    rng = np.random.default_rng(12345)
    randoms = rng.normal(0, 20, nobs).reshape(-1, 1)
    x = np.linspace(0, 100, nobs).reshape(-1, 1)
    y = 3*x + randoms

    ols = OLS(x, y)
    ols.fit()

    sm_X = sm.add_constant(x)
    sm_results = sm.OLS(y, sm_X).fit()
    aic, bic, hq = calc_IC_metrics(ols, normalize=False, verbose=False)
    ll = calc_log_likelihood(ols)
    assert aic == sm_results.aic
    assert bic == sm_results.bic
    assert hq == sm_tools.eval_measures.hqic(ll, nobs, len(sm_results.params))
