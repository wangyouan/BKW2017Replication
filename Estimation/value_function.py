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
        self._transition_matrix = None
        self._investment_grid = None
        self._debt_grid = None
        self._debt_policy_grid = None

        self._firm_value = None
        self.initialize(delta, rho, mu, gamma, theta, sigma, lambda_)

    def initialize(self, delta=None, rho=None, mu=None, gamma=None, theta=None, sigma=None, lambda_=None):
        self._delta = delta if delta is not None else parameters.delta_
        self._rho = rho if rho is not None else parameters.rho_
        self._mu = mu if mu is not None else parameters.mu_
        self._gamma = gamma if gamma is not None else parameters.gamma_
        self._theta = theta if theta is not None else parameters.theta_
        self._sigma = sigma if sigma is not None else parameters.sigma_
        self._lambda = lambda_ if lambda_ is not None else parameters.lambda_
        self._profitability_grid, self._transition_matrix = generate_profitability_distribution(self._mu, self._rho,
                                                                                                self._sigma, self.Z_NUM)

        self._debt_grid = get_range(-self._theta, self._theta, self.P_NUM)
        self._investment_grid = get_range(self._delta * (2 - np.ceil(self.I_NUM / (2 * self.DELP))),
                                          self._delta * (2 + np.ceil(self.I_NUM / (2 * self.DELP))),
                                          self.I_NUM)
        self._debt_policy_grid = get_range(-self._theta, self._theta, self.P_NEXT_NUM)
        self._firm_value = np.ones((self.P_NUM, self.Z_NUM))

    def optimize(self):
        firm_value = self._firm_value.copy()
        for _ in range(self.MAX_ITERATION):
            value_transpose = np.dot(firm_value, self._transition_matrix)
            value_transpose_next = np.zeros((self.P_NEXT_NUM, self.Z_NUM))
            for ip in range(self.P_NEXT_NUM):
                ipd = int(ip * self.P_NUM / self.P_NEXT_NUM)
                ipu = ipd + 1

                if ipu >= self.P_NUM:
                    ipd = self.P_NUM - 1
                    value_transpose_next[ip, :] = value_transpose[ipd, :]
                else:
                    value_transpose_next[ip, :] = value_transpose[ipd, :] + (
                            value_transpose[ipu, :] - value_transpose[ipd, :]) / (
                                                          self._debt_grid[ipu] - self._debt_grid[ipd]) * (
                                                          self._debt_policy_grid[ip] - self._debt_grid[ipd])
            queue_i = np.zeros((self.P_NEXT_NUM, self.I_NUM, self.Z_NUM))
            for i in range(self.I_NUM):
                queue_i[:, i, :] = value_transpose_next * (1 - self._delta + self._investment_grid[i])

            for i in range(self.P_NUM):
                pass

        else:
            raise RuntimeError('Model doesn\'t converge')

    def get_debt_policy(self):
        pass

    def get_payoff(self, profitability, investment, debt, next_debt):
        payoff = profitability - investment - 0.5 * self._gamma * investment ** 2 - debt * (1 + self.RF) + next_debt * (
                1 - self._delta + investment)

        if payoff < 0:
            payoff *= (1 + self._lambda)

        return payoff
