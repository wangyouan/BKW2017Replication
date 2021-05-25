#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Filename: value_function
# @Date: 2021/5/25
# @Author: Mark Wang
# @Email: markwang@connect.hku.hk
import numpy as np

from Constants.constants import Constants
from Constants.parameters import Parameters as parameters
from Utilities import generate_profitability_distribution, get_range


class FirmValue(Constants):
    """
    This function used to find the optimal value of firm value given
    """

    def __init__(self, delta=None, rho=None, mu=None, gamma=None, theta=None, sigma=None, lambda_=None):
        """
        init

        :param delta: depreciation rate.
        :param rho: autocorrelation of profitability shock,
        :param mu: drift of profitability shock
        :param gamma: adjustment cost of investment.
        :param theta: collateral parameter
        :param sigma: std. dev. of profitability shocks,
        :param lambda_: equity issuance cost.
        :return: value function and firm
        """
        self._delta = None
        self._rho = None
        self._mu = None
        self._gamma = None
        self._theta = None
        self._sigma = None
        self._lambda = None
        self._profitability_grid = None
        self._investment_grid = None
        self._debt_grid = None

        self.initialize(delta, rho, mu, gamma, theta, sigma, lambda_)

    def initialize(self, delta=None, rho=None, mu=None, gamma=None, theta=None, sigma=None, lambda_=None):
        self._delta = delta if delta is not None else parameters.delta_
        self._rho = rho if rho is not None else parameters.rho_
        self._mu = mu if mu is not None else parameters.mu_
        self._gamma = gamma if gamma is not None else parameters.gamma_
        self._theta = theta if theta is not None else parameters.theta_
        self._sigma = sigma if sigma is not None else parameters.sigma_
        self._lambda = lambda_ if lambda_ is not None else parameters.lambda_
        self._profitability_grid = generate_profitability_distribution(self._mu, self._rho, self._sigma, self.Z_NUM)

        self._debt_grid = get_range(-self._theta, self._theta, self.P_NUM)
        self._investment_grid = get_range(self._delta * (2 - np.ceil(self.I_NUM / (2 * self.DELP))),
                                          self._delta * (2 + np.ceil(self.I_NUM / (2 * self.DELP))),
                                          self.I_NUM)
