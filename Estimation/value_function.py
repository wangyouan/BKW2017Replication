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
from Utilities import generate_profitability_distribution, get_range, inter_product


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
        self._debt_policy_matrix = np.zeros((self.P_NUM, self.Z_NUM), dtype=int)
        self._invest_policy_matrix = np.zeros((self.P_NUM, self.Z_NUM), dtype=int)

    def optimize(self):
        """
        Optimize model from BKW 2018 crs_toni_v2.f90
        :return: error code list
            0: Succeed
            1: Don't coverage
            2: Too large model difference
        """
        firm_value = self._firm_value.copy()
        all_val = np.zeros((self.P_NUM, self.P_NEXT_NUM, self.I_NUM, self.Z_NUM))
        queue_i = np.zeros((self.P_NEXT_NUM, self.I_NUM, self.Z_NUM))
        payoff = self.get_payoff_matrix()

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
                    frac = (self._debt_grid[ipu] - self._debt_prime_grid[ip]) / (
                                self._debt_grid[ipu] - self._debt_grid[ipd])
                    print(frac)
                    expected_fv_prime[ip, :] = expected_fv[ipd, :] * frac + expected_fv[ipu, :] * (1 - frac)

            for i in range(self.I_NUM):
                queue_i[:, i, :] = expected_fv_prime * (1 - self._delta + self._investment_grid[i])

            for ip in range(self.P_NUM):
                all_val[ip, :, :, :] = payoff[ip, :, :, :] + self.BETA * queue_i

            firm_value_next = np.max(np.max(all_val, axis=1), axis=1)
            difference = firm_value_next - firm_value
            model_diff = np.max(np.abs(difference))
            if _ % 100 == 0:
                print('Iteration: %d, Model difference: %f' % (_, model_diff))
            if model_diff < self.COVERAGE_THRESHOLD:
                break
            elif model_diff > self.DIFF_MAX_THRESHOLD:
                return 2
                # break

            firm_value = firm_value_next.copy()

        else:
            return 1

        self._firm_value = firm_value.copy()

        for iz in range(self.Z_NUM):
            for ip in range(self.P_NUM):
                current_value_firm = all_val[ip, :, :, iz]
                debt_max_i, invest_max_i = np.unravel_index(np.argmax(current_value_firm, axis=None),
                                                            current_value_firm.shape)
                self._debt_policy_matrix[ip, iz] = debt_max_i
                self._invest_policy_matrix[ip, iz] = invest_max_i
                return 0

    def optimize_terry(self):
        """
        Follow Stephen James Terry. Junior Accounting Theory Workshop, intro to structural estimation, 2018

        :return: same as optimize
        """
        debt_policy_matrix = self._debt_policy_matrix.copy()
        invest_policy_matrix = self._invest_policy_matrix.copy()
        firm_value = self._firm_value.copy()
        firm_value_last = firm_value.copy()

        payoff = self.get_payoff_matrix()

        for _ in range(self.MAX_ITERATION):
            # value function iteration
            for __ in range(self.VALUE_FUNCTION_ITERATION):
                for ip in range(self.P_NUM):
                    for iz in range(self.Z_NUM):
                        invest = self._investment_grid[invest_policy_matrix[ip, iz]]
                        debt_prime = self._debt_prime_grid[debt_policy_matrix[ip, iz]]
                        cfrac, debt_low_i, debt_up_i = inter_product(debt_prime, self._debt_grid)
                        fv_prime = firm_value[debt_low_i, :] * cfrac + firm_value[debt_up_i, :] * (1 - cfrac)
                        current_payoff = payoff[ip, debt_policy_matrix[ip, iz], invest_policy_matrix[ip, iz], iz]
                        firm_value[ip, iz] = current_payoff + self.BETA * np.dot(
                            fv_prime, self._transition_matrix[iz, :]) * (1 - self._delta + invest)

            # update policy matrix
            for ip in range(self.P_NUM):
                for iz in range(self.Z_NUM):
                    firm_value_prob = np.zeros((self.P_NEXT_NUM, self.I_NUM))
                    for ip_prime in range(self.P_NEXT_NUM):
                        for ii in range(self.I_NUM):
                            debt_prime = self._debt_prime_grid[ip_prime]

                            cfrac, debt_low_i, debt_up_i = inter_product(debt_prime, self._debt_grid)
                            fv_prime = firm_value[debt_low_i, :] * cfrac + firm_value[debt_up_i, :] * (1 - cfrac)
                            firm_value_prob[ip_prime, ii] = payoff[ip, ip_prime, ii, iz] + self.BETA * np.dot(
                                fv_prime, self._transition_matrix[iz, :]) * (1 - self._delta + self._investment_grid[
                                ii])

                    debt_max_i, invest_max_i = np.unravel_index(np.argmax(firm_value_prob, axis=None),
                                                                firm_value_prob.shape)
                    debt_policy_matrix[ip, iz] = debt_max_i
                    invest_policy_matrix[ip, iz] = invest_max_i

            # check difference
            value_difference = firm_value - firm_value_last
            diff = np.max(np.abs(value_difference))
            if _ % 10 == 0:
                print('Iteration: %d, Model difference: %f' % (_, diff))
            if diff > self.DIFF_MAX_THRESHOLD:
                return 2
            elif diff < self.COVERAGE_THRESHOLD:
                self._firm_value = firm_value.copy()
                self._debt_policy_matrix = debt_policy_matrix.copy()
                self._invest_policy_matrix = invest_policy_matrix.copy()
                return 0

            firm_value_last = firm_value.copy()

        else:
            # not coverage
            return 1

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

    def get_payoff_matrix(self):
        payoff = np.zeros((self.P_NUM, self.P_NEXT_NUM, self.I_NUM, self.Z_NUM))

        for ip in range(self.P_NUM):
            for ip_prime in range(self.P_NEXT_NUM):
                for ii in range(self.I_NUM):
                    for iz in range(self.Z_NUM):
                        payoff[ip, ip_prime, ii, iz] = self.get_payoff(
                            self._profitability_grid[iz], self._investment_grid[ii], self._debt_grid[ip],
                            self._debt_prime_grid[ip_prime])
        return payoff

    def get_payoff(self, profitability, investment, debt, next_debt):
        payoff = profitability - investment - 0.5 * self._gamma * (investment ** 2) - debt * (1 + self.RF) \
                 + next_debt * (1 - self._delta + investment)

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
        investment_array = [self._investment_grid[self._invest_policy_matrix[debt_index[i], profit_index[i]]] for i in
                            range(firms)]
        debt_array = [self._debt_grid[int(i * self.P_NUM)] for i in init_value]
        trans_cdf = self._transition_matrix.copy()

        for i in range(self.Z_NUM - 2):
            trans_cdf[:, i + 1] += trans_cdf[:, i]

        trans_cdf[:, self.Z_NUM - 1] = 1

        # run simulation
        simulated_data_list = list()
        for year_i in range(years):
            simulated_data = DataFrame(
                columns=['value', 'profitability', 'debt', 'investment', 'payoff', 'debt_prime'],
                index=list(range(firms)))

            simulated_data.loc[:, 'profitability'] = self._profitability_grid[profit_index]
            simulated_data.loc[:, 'debt'] = debt_array
            simulated_data.loc[:, 'value'] = value_array

            # get next period profitability
            next_shock = np.random.random(firms)
            for i in range(firms):
                # save current data
                profitability = simulated_data.loc[i, 'profitability']
                debt = simulated_data.loc[i, 'debt']

                # determine next period values
                cfrac, low_index, up_index = inter_product(debt_array[i], self._debt_grid)
                profit_index[i] = len(trans_cdf[profit_index[i]][trans_cdf[profit_index[i]] < next_shock[i]])
                debt_prime_up = self._debt_prime_grid[self._debt_policy_matrix[up_index, profit_index[i]]]
                debt_prime_low = self._debt_prime_grid[self._debt_policy_matrix[low_index, profit_index[i]]]
                invest_prime_up = self._investment_grid[self._invest_policy_matrix[up_index, profit_index[i]]]
                invest_prime_low = self._investment_grid[self._invest_policy_matrix[low_index, profit_index[i]]]

                debt_array[i] = cfrac * debt_prime_low + (1 - cfrac) * debt_prime_up
                value_array[i] = cfrac * self._firm_value[low_index, profit_index[i]] + (1 - cfrac) \
                                 * self._firm_value[up_index, profit_index[i]]
                investment_array[i] = cfrac * invest_prime_low + (1 - cfrac) * invest_prime_up

                simulated_data.loc[i, 'payoff'] = self.get_payoff(profitability, investment_array[i], debt,
                                                                  debt_array[i])

            simulated_data.loc[:, 'year'] = year_i
            simulated_data.loc[:, 'firm_id'] = list(range(firms))
            simulated_data.loc[:, 'debt_prime'] = debt_array
            simulated_data.loc[:, 'investment'] = investment_array
            simulated_data_list.append(simulated_data)

        return pd.concat(simulated_data_list, ignore_index=True, sort=False)


if __name__ == '__main__':
    # test model value
    mu, rho, sigma, delta, gamma, theta, lambda_ = -2.2067, 0.8349, 0.3594, 0.0449, 29.9661, 0.3816, 0.1829
    fv = FirmValue(delta=delta, rho=rho, mu=mu, gamma=gamma, theta=theta, sigma=sigma, lambda_=lambda_)
    error_code = fv.optimize()
    sim_data = fv.simulate_model(58, 1)

    # test generate payoff function
    # theta = 0.4177055908170345
    # delta = 0.0634630044682483
    # rho = 0.5286754702975660
    # mu = -2.2413654287674789
    # sigma = 0.5457619195960740
    # gamma = 41.0530988801549661
    # payoff = np.zeros((FirmValue.P_NUM, FirmValue.P_NEXT_NUM, FirmValue.I_NUM, FirmValue.Z_NUM))
    # debt_grid = get_range(-theta, theta, FirmValue.P_NUM)
    # investment_grid = get_range(delta * (2 - np.ceil(FirmValue.I_NUM / (2 * FirmValue.DELP))),
    #                             delta * (2 - np.ceil(FirmValue.I_NUM / (2 * FirmValue.DELP))
    #                                      + (FirmValue.I_NUM - 1) / FirmValue.DELP),
    #                             FirmValue.I_NUM)
    # debt_prime_grid = get_range(-theta, theta, FirmValue.P_NEXT_NUM)
    # profitability_grid, transition_matrix = generate_profitability_distribution(mu, rho, sigma, FirmValue.Z_NUM)
    # for ip in range(FirmValue.P_NUM):
    #     for ip_prime in range(FirmValue.P_NEXT_NUM):
    #         for ii in range(FirmValue.I_NUM):
    #             for iz in range(FirmValue.Z_NUM):
    #                 payoff[ip, ip_prime, ii, iz] = profitability_grid[iz] - investment_grid[ii] - \
    #                                                0.5 * gamma * investment_grid[ii] ** 2 - debt_grid[ip] \
    #                                                * (1 + FirmValue.RF) + debt_prime_grid[ip_prime] * (
    #                                                        1 - delta + investment_grid[ii])
