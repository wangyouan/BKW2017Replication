#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Filename: value_function_smm_school
# @Date: 2021/7/28
# @Author: Mark Wang
# @Email: markwang@connect.hku.hk

import numpy as np
import pandas as pd
from pandas import DataFrame
from quantecon import tauchen

# define some constants
NUM_PROFITABILITY = 15
NUM_CAPITAL = 250
CAPITAL_MIN = 0.5
CAPITAL_MAX = 250

MAX_ITERATION = 3000
CONVERAGE_THRESHOLD = 1e-5
MAX_DIFF_THRESHOLD = 1e8


class FirmValue(object):
    def __init__(self, alpha, delta):
        self._alpha = alpha
        self._delta = delta
        self._beta = 0.96
        self._rho = 0.75
        self._sigma = 0.3
        self._lambda = 0.05

        self._profitability = tauchen(self._rho, self._sigma, m=2, n=NUM_PROFITABILITY)
        self._profitability.state_values = np.exp(self._profitability.state_values)
        self._capital_grid = np.arange(CAPITAL_MIN, CAPITAL_MAX, (CAPITAL_MAX - CAPITAL_MIN) / NUM_CAPITAL,
                                       dtype=np.float32)
        self._firm_value = np.zeros((NUM_CAPITAL, NUM_PROFITABILITY))
        self._capital_policy_grid = np.zeros((NUM_CAPITAL, NUM_PROFITABILITY))

    def optimize(self):
        # initialize payout grid
        payout_grid = np.zeros((NUM_CAPITAL, NUM_CAPITAL, NUM_PROFITABILITY))
        for ik in range(NUM_CAPITAL):
            for ik_prime in range(NUM_CAPITAL):
                for iz in range(NUM_PROFITABILITY):
                    z = self._profitability.state_values[iz]
                    k = self._capital_grid[ik]
                    k_prime = self._capital_grid[ik_prime]
                    payout_grid[ik, ik_prime, iz] = z * \
                                                    k ** self._alpha - k_prime + (1 - self._delta) * k

        payout_grid = np.where(payout_grid > 0, payout_grid,
                               (1 + self._lambda) * payout_grid)

        firm_value = self._firm_value.copy()
        for _ in range(MAX_ITERATION):
            firm_value_prime = firm_value @ self._profitability.P

            all_firm_value = np.zeros((NUM_CAPITAL, NUM_CAPITAL, NUM_PROFITABILITY))
            for ik in range(NUM_CAPITAL):
                all_firm_value[ik, :, :] = payout_grid[ik, :, :] + self._beta * firm_value_prime

            new_firm_value = np.max(all_firm_value, axis=1)

            model_difference = np.max(np.abs(new_firm_value - firm_value))
            if model_difference > MAX_DIFF_THRESHOLD:
                print('Cannot converage')
                return 1
            elif model_difference < CONVERAGE_THRESHOLD:
                self._firm_value = new_firm_value.copy()
                self._capital_policy_grid = np.argmax(all_firm_value, axis=1)
                return 0

            firm_value = new_firm_value.copy()

        else:
            return 2

    def simulate_model(self, n_firms, n_years):
        np.random.seed(1000)
        initial_state = np.random.random((n_firms, 2))
        profit_shock = np.random.random((n_firms, n_years))
        profit_index = np.array([int(i * NUM_PROFITABILITY) for i in initial_state[:, 0]], dtype=np.int32)
        capital_index = np.array([int(i * NUM_CAPITAL) for i in initial_state[:, 1]], dtype=np.int32)
        value_array = np.array([self._firm_value[capital_index[i], profit_index[i]] for i in range(n_firms)],
                               dtype=np.float32)

        simulated_data_list = list()
        cdf = self._profitability.cdfs

        for year in range(n_years):
            current_trans = cdf[profit_index, :]
            simulated_data = DataFrame(
                columns=['firm_id', 'year', 'capital', 'inv_rate', 'profitability', 'firm_value'])
            simulated_data.loc[:, 'firm_value'] = value_array
            simulated_data.loc[:, 'capital'] = self._capital_grid[capital_index]
            simulated_data.loc[:, 'profitability'] = self._profitability.state_values[profit_index]
            capital_policy = np.array(
                [self._capital_policy_grid[capital_index[i], profit_index[i]] for i in range(n_firms)])
            investment = self._capital_grid[capital_policy] - (1 - self._delta) * simulated_data.loc[:, 'capital']
            simulated_data.loc[:, 'inv_rate'] = investment / simulated_data.loc[:, 'capital']

            simulated_data.loc[:, 'firm_id'] = list(range(n_firms))
            simulated_data.loc[:, 'year'] = year
            simulated_data_list.append(simulated_data)

            capital_index = capital_policy
            profit_shock_series = profit_shock[:, year]

            profit_index = np.array(
                [len(current_trans[i][current_trans[i] < profit_shock_series[i]]) for i in range(n_firms)])

        simulated_data_df: DataFrame = pd.concat(simulated_data_list, ignore_index=True, sort=False)
        return simulated_data_df


if __name__ == '__main__':
    fv = FirmValue(0.6, 0.04)
    error_code = fv.optimize()
    simulated_data = fv.simulate_model(1000, 58)
