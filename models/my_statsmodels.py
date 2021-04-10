import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from statsmodels.tsa.stattools import adfuller


class OLS():
    def __init__(self, X, Y, add_const=True):
        assert(X.shape[0] == Y.shape[0])
        assert(Y.shape[1] == 1) # only allow for single Y variable
        self._X = X
        self._Y = Y.squeeze()

        self._has_const = (X[:, 0] == 1.0).all()
        if (not self._has_const) and add_const:
            self._X = np.concatenate([np.ones((self._X.shape[0], 1)),
                                      self._X], axis=1)
            self._has_const = True

        self._n = self._X.shape[0]
        self._k = self._X.shape[1]
        self._df = self._n - self._k

        # calculate and store (X' * X)^-1 to save computation time
        self._XtXinv = np.linalg.inv(np.matmul(self._X.T, self._X))

    @property
    def beta(self):
        return self._beta.flatten()

    def fit(self, cov_type='ols'):
        self._calc_beta()
        self._calc_residuals()
        self._calc_std_errors(cov_type=cov_type)
        self._calc_R2()
        self._calc_PM()
        self._calc_cond_no()

        Eex = np.matmul(self._X.T, self.e) * (1 / self._n)
        self._Eex_is_zero = np.isclose(Eex, np.zeros(Eex.shape),
                                       atol=1e-12).all()

    def predict(self, X):
        if X.shape[-1] != self.beta.shape[0]:
            raise Exception('Shape mismatch. X.shape: {}. self.beta.shape: {}'.format(X.shape, self.beta.shape))
        return np.matmul(X, self.beta)

    def t_test(self, beta_null, verbose=True):
        t_value = (self.beta - beta_null) / self.std_errors

        results = {}
        alphas = [0.9, 0.95, 0.99]

        for alpha in alphas:
            min_val = -stats.t.ppf((1 + alpha) / 2, df=self._df)
            max_val = stats.t.ppf((1 + alpha) / 2, df=self._df)

            results[alpha] = ~((t_value < max_val) & (t_value > min_val))

        if verbose:
            print('T-Test Results'.ljust(47) + 'Reject at')
            print('beta_hat'.ljust(12)
                  + 'beta_null'.ljust(12)
                  + 't-value'.ljust(15)
                  + '{:.2f}?'.format(1 - alphas[0]).ljust(10)
                  + '{:.2f}?'.format(1 - alphas[1]).ljust(10)
                  + '{:.2f}?'.format(1 - alphas[2]).ljust(10))

            iterator = zip(self.beta, beta_null, t_value, results[alphas[0]],
                           results[alphas[1]], results[alphas[2]])
            for b_h, b_n, t, r1, r2, r3 in iterator:
                print('{: .6f}'.format(b_h).ljust(12)
                      + '{: .6f}'.format(b_n).ljust(12)
                      + '{: .6f}'.format(t).ljust(15)
                      + '{}'.format(r1).ljust(10)
                      + '{}'.format(r2).ljust(10)
                      + '{}'.format(r3).ljust(10))

        return results

    def conf_interval(self, alpha, verbose=True):
        results = {}

        stat = stats.t.ppf((1 + alpha) / 2, df=self._df)
        results['high'] = self.beta + stat * self.std_errors
        results['low'] = self.beta - stat * self.std_errors

        if verbose:
            print('{:.2f}% Confidence Intervals for Beta'.format(alpha * 100))
            print('beta_hat'.ljust(15)
                  + '{:.3f}'.format(1 - (1 + alpha) / 2).ljust(15)
                  + '{:.3f}'.format((1 + alpha) / 2).ljust(15))

        for beta, high, low in zip(self.beta, results['high'], results['low']):
            print('{: .6f}'.format(beta).ljust(15)
                  + '{: .6f}'.format(low).ljust(15)
                  + '{: .6f}'.format(high).ljust(15))

        return results

    def print_results(self):
        print('OLS Regression Results')
        print('beta:')
        for i, b in enumerate(self.beta):
            if self._has_const and (i == 0):
                print('- constant:\t {: .6f}'.format(b))
            else:
                print('- beta_{}:\t {: .6f}'.format(i, b))

        if not self._has_const:
            print('Warning: The model does not include a constant term.')

        stder_type = 'OLS' if self._cov_type == 'ols' else 'White'
        print('{} Standard Errors for beta:'.format(stder_type))
        for i, err in enumerate(self.std_errors):
            if self._has_const and (i == 0):
                print('- constant:\t {: .6f}'.format(err))
            else:
                print('- std_error_{}:\t {: .6f}'.format(i, err))

        print('R2: {:.6f}'.format(self.R2))
        print('Adjusted R2: {:.6f}'.format(self.adjusted_R2))

        if self._PM_is_zero:
            print('PM equals zero')
        else:
            print('PM does not equal zero')

        if self._Eex_is_zero:
            print('E[ex] equals zero')
        else:
            print('E[ex] does not equal zero')

        print('Condition No. of second moment matrix: {:.6f}'.format(self.k))
        if self._ill_conditioned:
            print('Warning: The second moment matrix is ill-conditioned.'
                  + 'Precision will be lost')

    def _calc_beta(self):
        self._beta = np.matmul(self._XtXinv, np.matmul(self._X.T, self._Y))

    def _calc_residuals(self):
        self.e = (self._Y - self.predict(self._X)).squeeze()

    def _calc_std_errors(self, cov_type='ols'):
        self._cov_type = cov_type
        self._sigma_2_hat = 1 / (self._n - self._k) * np.dot(self.e, self.e)

        if cov_type == 'ols':
            self._cov = self._XtXinv * self._sigma_2_hat

        elif cov_type == 'white':
            D = np.zeros((self._k, self._k))
            for i in range(self._n):
                D += np.outer(self._X[i, :], self._X[i, :]) * self.e[i]**2

            self._cov = ((self._n / (self._n - self._k))
                         * np.matmul(np.matmul(self._XtXinv, D), self._XtXinv))

        else:
            raise ValueError('{} is not a valid cov_type'.format(cov_type))

        self.std_errors = np.sqrt(np.diagonal(self._cov))

    def _calc_R2(self):
        sigma_2_hat_y = (1 / self._n) * np.dot(self._Y - self._Y.mean(),
                                               self._Y - self._Y.mean())

        self.R2 = 1 - (1 / self._n) * np.dot(self.e, self.e) / sigma_2_hat_y
        self.adjusted_R2 = 1 - ((1 - self.R2) *
                            ((self._n - 1) / (self._n - self._k)))

    def _calc_PM(self):
        self.P = np.matmul(np.matmul(self._X, self._XtXinv), self._X.T)
        self.M = np.eye(self._n) - self.P

        PM = np.matmul(self.P, self.M)
        self._PM_is_zero = np.isclose(PM, np.zeros(PM.shape), atol=1e-12).all()

    def _calc_cond_no(self):
        Qxx = np.matmul(self._X.T, self._X) * (1 / self._n)
        self.k = np.linalg.cond(Qxx)

        # As a rule of thumb, if the condition number k(A)=10^k then you may
        # lose up to k digits of accuracy
        eps = 1 / 1e-16 # for accuracy up to 16 digits of precision
        self._ill_conditioned = (self.k > eps)


class AR1():
    # class for an AR(1) process with normal white noise
    def __init__(self, phi, eps_mu=0, eps_var=1, seed=None):
        self.phi = phi
        self.eps_mu = eps_mu
        self.eps_var = eps_var
        self.seed(seed)

    def sample(self, samples, start=0):
        x_prev = start
        sample = []
        for _ in range(samples):
            eps = self._rng.normal(self.eps_mu, self.eps_var, 1)[0]
            x = self.phi * x_prev + eps
            sample.append(x)
            x_prev = x

        return np.array(sample)

    def seed(self, seed):
        self._rng = np.random.default_rng(seed)


def calc_IC_metrics(regression, normalize=True, verbose=True):
    sigma_hat = np.sqrt(1 / (regression._n - ols._k)
                        * np.dot(regression.e, regression.e))
    log_likelihood = sum(np.log(stats.norm.pdf(regression.e, loc=0.0,
                                               scale=sigma_hat)))

    # https://en.wikipedia.org/wiki/Akaike_information_criterion
    aic = -2 * log_likelihood + 2 * regression._k

    # https://en.wikipedia.org/wiki/Bayesian_information_criterion
    bic = -2 * log_likelihood + regression._k * np.log(regression._n)

    # from slides (without normalization)
    hq = -2 * log_likelihood + 2 * regression._k * np.log(np.log(regression._n))

    if normalize:
        aic = aic / regression._n
        bic = bic / regression._n
        hq = hq / regression._n

    if verbose:
        print('AIC: {: .4f}'.format(aic))
        print('BIC: {: .4f}'.format(bic))
        print('HQ IC: {: .4f}'.format(hq))

    return aic, bic, hq


def durbin_watson(regression, verbose=True):
    # using the formula here, https://en.wikipedia.org/wiki/Durbin%E2%80%93Watson_statistic,
    # which starts the summation in the denominator at t=1 instead of t=2
    # like the slides
    dw = (sum((regression.e[:-1] - regression.e[1:])**2)
          / sum(regression.e**2))

    if verbose:
        print('Durbin-Watson: {: .4f}'.format(dw))

    return dw


def breusch_godfrey(regression, r, include_regressors=True, verbose=True):
    # data should be in
    assert(r >= 1)

    Y = regression.e[r:].reshape(-1, 1)
    X = np.concatenate([regression.e[r-i:-i].reshape(-1, 1)
                        for i in range(1, r + 1)],
                       axis=1)

    if include_regressors:
        X = np.concatenate([regression._X[r:], X], axis=1)

    # only adds a constant if the original X doesn't have one
    aux_regression = OLS(X, Y, add_const=True)
    aux_regression.fit()

    test_stat = (aux_regression._n) * aux_regression.R2
    p_value = 1.0 - stats.chi2.cdf(test_stat, df=r)

    if verbose:
        print('Test Statistic: {: .4f} | P-Value: {: .4f}'.format(test_stat,
                                                                  p_value))

    return aux_regression, test_stat, p_value


def residual_qq_plot(regression):
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(1, 1, 1)

    x = []
    y = []
    for q in np.linspace(0.0, 1.0, 100):
        x.append(np.quantile(regression.e, q))
        y.append(stats.norm.ppf(q, loc=0.0,
            scale=np.sqrt(1 / (regression._n - regression._k)
                          * np.dot(regression.e, regression.e))))

    ax.scatter(x, y)
    ax.set_title('QQ Residuals Plot')
    ax.set_xlabel('Quantile of Residuals')
    ax.set_ylabel('Qunatile of Normal')

    x = np.linspace(*ax.get_xlim())
    ax.plot(x, x, color='orange')

    plt.show()


def roll_X(X, lags=1):
    X = np.stack([np.roll(X, l) for l in range(lags + 1)])
    X = X[1:, lags:].T

    return X


def calc_acf(data, lags=20):
    assert data.ndim == 1, 'Multi-dimensional data not supported'
    assert len(data) > lags, 'Data must be longer than the number of lags'
    acf = [1.0]
    if lags == 0:
        return np.array(acf)

    mean = np.mean(data)
    var = np.var(data, ddof=1)
    n = data.shape[0]

    for lag in range(1, lags + 1):
        # calculate covariance with the mean and variance of
        # the entire series rather than use np.corrcoef or np.cov
        # similar to the statsmodels approach
        cov = (1 / (n - 1)) * np.matmul((data[:-lag] - mean),
                                        (data[lag:] - mean))
        ac = cov / var
        acf.append(ac)

    return np.array(acf)


def calc_pacf(data, lags=20):
    assert data.ndim == 1, 'Multi-dimensional data not supported'
    assert len(data) > lags, 'Data must be longer than the number of lags'
    pacf = [1.0]
    if lags == 0:
        return np.array(pacf)

    for lag in range(1, lags + 1):
        y = data[lag:].reshape(-1, 1)
        X = roll_X(data, lags=lag)
        ols = OLS(X, y, add_const=True)
        ols._calc_beta()
        pacf.append(ols.beta[-1])

    return np.array(pacf)


def plot_acf(data, lags=20, alpha=0.05, ax=None):
    n = data.shape[0]
    acf = calc_acf(data, lags=lags)
    std_error = np.sqrt((np.insert(1 + 2 * np.cumsum(acf ** 2), 0, 1,
                                   axis=0)[:-1]) / n)
    upper_ci = std_error * stats.norm.ppf(1 - alpha / 2)
    lower_ci = std_error * stats.norm.ppf(alpha / 2)

    if ax is None:
        fig, ax = plt.subplots()

    x_axis = np.array(range(lags + 1))

    ax.scatter(x_axis, acf, zorder=999) # zorder to move the dots to the front
    ax.vlines(x_axis, 0, acf, color='k')
    ax.fill_between(x=(x_axis + 0.5), y1=lower_ci, y2=upper_ci,
                    facecolor='b', alpha=0.2)
    ax.set_title('Autocorrelation')


def plot_pacf(data, lags=20, alpha=0.05, ax=None):
    n = data.shape[0]
    pacf = calc_pacf(data, lags=lags)
    upper_ci = stats.norm.ppf(1 - alpha / 2) / np.sqrt(n)
    lower_ci = stats.norm.ppf(alpha / 2) / np.sqrt(n)

    if ax is None:
        fig, ax = plt.subplots()

    x_axis = np.array(range(lags + 1))

    ax.scatter(x_axis, pacf, zorder=999) # zorder to move the dots to the front
    ax.vlines(x_axis, 0, pacf, color='k')
    ax.fill_between(x=(x_axis + 0.5), y1=lower_ci, y2=upper_ci,
                    facecolor='b', alpha=0.2)
    ax.set_title('Partial Autocorrelation')


def df_test(data, verbose=True):
    assert data.ndim == 1, 'Multi-dimensional data not supported'

    y = np.diff(data).reshape(-1, 1)
    X = data[:-1].reshape(-1, 1)

    ols = OLS(X, y, add_const=False)
    ols.fit()

    t_stat = ols.beta[0] / ols.std_errors[0]
    adf_results = adfuller(data, maxlag=0, regression='nc')
    assert(abs(adf_results[0] - t_stat) < 1e-6) # check the t-stat

    results = {
        't_stat': t_stat,
        'critical_values': adf_results[4]
    }

    if verbose:
        print('Dickey-Fuller Test')
        print('t-statistic: {:.6f}'.format(t_stat))
        print('Critical Values')
        for p, v in results['critical_values'].items():
            print('{}: {:.6f}'.format(p, v))

    return results


def adf_test(data, lags=0, regression='nc', verbose=True):
    assert (data.ndim == 1), 'Multi-dimensional data not supported'
    assert (regression in ['c', 'ct', 'nc'])

    y = np.diff(data).reshape(-1, 1)
    X = data[:-1].reshape(-1, 1)

    lagged_diffs = roll_X(y.flatten(), lags=lags)
    X = np.concatenate([X[lags: ], lagged_diffs], axis=1)

    if regression == 'ct':
        t = np.array(range(1, X.shape[0] + 1)).reshape(-1, 1)
        X = np.concatenate([X, t], axis=1)

    y = y[lags: ]

    add_const = (regression == 'c' or regression == 'ct')
    ols = OLS(X, y, add_const=add_const)
    ols.fit()

    beta_index = (1 if add_const else 0)
    t_stat = ols.beta[beta_index] / ols.std_errors[beta_index]
    adf_results = adfuller(data, maxlag=lags,
                           regression=regression, autolag=None)
    assert(abs(adf_results[0] - t_stat) < 1e-6) # check the t-stat

    results = {
        't_stat': t_stat,
        'critical_values': adf_results[4]
    }

    if verbose:
        print('Augmented Dickey-Fuller Test')
        print('t-statistic: {:.6f}'.format(t_stat))
        print('Critical Values')
        for p, v in results['critical_values'].items():
            print('{}: {:.6f}'.format(p, v))

    return results
