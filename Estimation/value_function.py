#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Filename: value_function
# @Date: 2021/5/25
# @Author: Mark Wang
# @Email: markwang@connect.hku.hk

import pandas as pd
from pandas import DataFrame
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
        self._debt_prime_grid = None

        self._firm_value = None

        # store policy matrix
        self._debt_policy_matrix = None
        self._invest_policy_matrix = None

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
        self._debt_prime_grid = get_range(-self._theta, self._theta, self.P_NEXT_NUM)
        self._firm_value = np.zeros((self.P_NUM, self.Z_NUM))
        self._debt_policy_matrix = np.zeros((self.P_NUM, self.Z_NUM))
        self._invest_policy_matrix = np.zeros((self.P_NUM, self.Z_NUM))

    def optimize(self):
        """
        Optimize model
        :return: error code list
            0: Succeed
            1: Don't coverage
            2: Too large model difference
        """
        firm_value = self._firm_value.copy()
        all_val = np.zeros((self.P_NUM, self.P_NEXT_NUM, self.I_NUM, self.Z_NUM))
        queue_i = np.zeros((self.P_NEXT_NUM, self.I_NUM, self.Z_NUM))
        payoff = np.zeros((self.P_NUM, self.P_NEXT_NUM, self.I_NUM, self.Z_NUM))

        for ip in range(self.P_NUM):
            for ip_prime in range(self.P_NEXT_NUM):
                for ii in range(self.I_NUM):
                    for iz in range(self.Z_NUM):
                        payoff[ip, ip_prime, ii, iz] = self.get_payoff(
                            self._profitability_grid[iz], self._investment_grid[ii], self._debt_grid[ip],
                            self._debt_prime_grid[ip_prime])

        for _ in range(self.MAX_ITERATION):
            expected_fv = np.dot(firm_value, self._transition_matrix)
            expected_fv_prime = np.zeros((self.P_NEXT_NUM, self.Z_NUM))
            for ip in range(self.P_NEXT_NUM):
                ipd = int(ip * (self.P_NUM - 1) / (self.P_NEXT_NUM - 1))
                ipu = ipd + 1

                if ipu >= self.P_NUM:
                    ipd = self.P_NUM - 1
                    expected_fv_prime[ip, :] = expected_fv[ipd, :]
                else:
                    expected_fv_prime[ip, :] = (expected_fv[ipd, :] * (
                            self._debt_grid[ipu] - self._debt_prime_grid[ip]) + expected_fv[ipu, :] * (
                                                           self._debt_prime_grid[ip] - self._debt_grid[ipd])) / (
                                                          self._debt_grid[ipu] - self._debt_grid[ipd])

            for i in range(self.I_NUM):
                queue_i[:, i, :] = expected_fv_prime * (1 - self._delta + self._investment_grid[i])

            for ip in range(self.P_NUM):
                all_val[ip, :, :, :] = payoff[ip, :, :, :] + self.BETA * queue_i

            firm_value_next = np.max(np.max(all_val, axis=2), axis=1)
            difference = firm_value_next - firm_value
            model_diff = abs(np.max(difference))
            if _ % 100 == 0:
                print('Iteration: %d, Model difference: %f' % (_, model_diff))
            if model_diff < self.COVERAGE_THRESHOLD:
                break
            elif model_diff > self.DIFF_MAX_THRESHOLD:
                return 2

            firm_value = firm_value_next.copy()

        else:
            return 1

        self._firm_value = firm_value.copy()

        for iz in range(self.Z_NUM):
            for ip in range(self.P_NUM):
                current_value_firm = all_val[ip, :, :, iz]
                self._debt_policy_matrix[ip, iz] = self._debt_prime_grid[np.argmax(np.max(current_value_firm, axis=1))]
                self._invest_policy_matrix[ip, iz] = self._investment_grid[
                    np.argmax(np.max(current_value_firm, axis=0))]
        return 0

    def set_model_parameters(self, delta=None, rho=None, mu=None, gamma=None, theta=None, sigma=None, lambda_=None):
        if delta is not None:
            self.set_delta(delta)

        if rho is not None:
            self.set_rho(rho)

        if mu is not None:
            self.set_mu(mu)

        if gamma is not None:
            self.set_gamma(gamma)

        if theta is not None:
            self.set_theta(theta)

        if sigma is not None:
            self.set_sigma(sigma)

        if lambda_ is not None:
            self.set_lambda(lambda_)

    def set_delta(self, delta):
        self._delta = delta

    def set_rho(self, rho):
        self._rho = rho

    def set_mu(self, mu):
        self._mu = mu

    def set_gamma(self, gamma):
        self._gamma = gamma

    def set_theta(self, theta):
        self._theta = theta

    def set_sigma(self, sigma):
        self._sigma = sigma

    def set_lambda(self, lambda_):
        self._lambda = lambda_

    def get_payoff(self, profitability, investment, debt, next_debt):
        payoff = profitability - investment - 0.5 * self._gamma * investment ** 2 - debt * (1 + self.RF) + next_debt * (
                1 - self._delta + investment)

        if payoff < 0:
            payoff *= (1 + self._lambda)

        return payoff

    def simulate_model(self, years, firms):
        """
        simulate model
        :param years: number of year
        :param firms: number of firms
        :return: simulated model vectors
            firm_id
            year
            value
            profitability
            debt
            investment
            payoff
        """
        # initialize
        init_value = np.random.random(firms)
        profit_index = [int(i * self.Z_NUM) for i in init_value]
        debt_index = [int(i * self.P_NUM) for i in init_value]
        value_array = [self._firm_value[debt_index[i], profit_index[i]] for i in range(firms)]
        investment_array = [self._invest_policy_matrix[debt_index[i], profit_index[i]] for i in range(firms)]
        debt_array = [self._debt_grid[int(i * self.P_NUM)] for i in init_value]
        trans_cdf = self._transition_matrix.copy()

        for i in range(self.Z_NUM - 2):
            trans_cdf[:, i + 1] += trans_cdf[:, i]

        trans_cdf[:, self.Z_NUM - 1] = 1

        # run simulation
        simulated_data_list = list()
        for year_i in range(years):
            simulated_data = DataFrame(
                columns=['value', 'profitability', 'debt', 'investment', 'payoff'], index=list(range(firms)))

            simulated_data.loc[:, 'profitability'] = self._profitability_grid[profit_index]
            simulated_data.loc[:, 'debt'] = debt_array
            simulated_data.loc[:, 'value'] = value_array
            simulated_data.loc[:, 'investment'] = investment_array

            # get next period profitability
            next_shock = np.random.random(firms)
            for i in range(firms):
                # save current data
                profitability = simulated_data.loc[i, 'profitability']
                investment = simulated_data.loc[i, 'investment']
                debt = simulated_data.loc[i, 'debt']

                # determine next period values
                debt_index = max(len(self._debt_grid[self._debt_grid < debt_array[i]]) - 1, 0)
                profit_index[i] = len(trans_cdf[profit_index[i]][trans_cdf[profit_index[i]] < next_shock[i]])
                if debt_index == 0:
                    debt_array[i] = self._debt_policy_matrix[debt_index, profit_index[i]]
                    value_array[i] = self._firm_value[debt_index, profit_index[i]]
                    investment_array[i] = self._invest_policy_matrix[debt_index, profit_index[i]]

                else:
                    fraction = (debt_array[i] - self._debt_grid[debt_index - 1]) * (
                            self._debt_grid[debt_index] - self._debt_grid[debt_index - 1])
                    debt_array[i] = fraction * self._debt_policy_matrix[debt_index, profit_index[i]] + (1 - fraction) \
                                    * self._debt_policy_matrix[debt_index - 1, profit_index[i]]
                    value_array[i] = fraction * self._firm_value[debt_index, profit_index[i]] + (1 - fraction) \
                                     * self._firm_value[debt_index - 1, profit_index[i]]
                    investment_array[i] = fraction * self._invest_policy_matrix[debt_index, profit_index[i]] + \
                                          (1 - fraction) * self._invest_policy_matrix[debt_index - 1, profit_index[i]]

                simulated_data.loc[i, 'payoff'] = self.get_payoff(profitability, investment, debt, debt_array[i])

            simulated_data.loc[:, 'year'] = year_i
            simulated_data.loc[:, 'firm_id'] = list(range(firms))
            simulated_data_list.append(simulated_data)

        return pd.concat(simulated_data_list, ignore_index=True, sort=False)


if __name__ == '__main__':
    fv = FirmValue(delta=0.0449, rho=0.8, mu=-2.4, gamma=40, theta=0.4, sigma=0.3, lambda_=0.2)
    error_code = fv.optimize()
    sim_data = fv.simulate_model(58, 900)
