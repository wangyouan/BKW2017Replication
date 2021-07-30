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

from EstimationSummerSchool.numba_method import optimize, simulate_model, optimizeinv
from EstimationSummerSchool import NUM_PROFITABILITY, NUM_CAPITAL, CAPITAL_MAX, CAPITAL_MIN, NUM_INVESTMENT


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
        error_code, firm_value, all_firm_value = optimize(self._alpha, self._delta, self._lambda, self._beta,
                                                          self._profitability.state_values, self._profitability.P,
                                                          self._capital_grid, self._firm_value)
        if error_code == 0:
            self._firm_value = firm_value.copy()
            self._capital_policy_grid = np.argmax(all_firm_value, axis=1)

        return error_code

    def simulate_model(self, n_firms, n_years):
        simulated_results = simulate_model(self._delta, n_firms, n_years, self._firm_value,
                                           self._profitability.state_values, self._profitability.cdfs,
                                           self._capital_grid, self._capital_policy_grid)
        simulated_result = np.vstack(simulated_results)

        simulated_df = DataFrame(simulated_result,
                                 columns=['firm_id', 'year', 'capital', 'investment', 'inv_rate', 'profitability',
                                          'value'])
        simulated_df.loc[:, 'inv_rate'] = simulated_df['investment'] / simulated_df.loc[:, 'capital']
        simulated_df.loc[:, 'profitability'] *= simulated_df.loc[:, 'capital'].apply(lambda x: x ** (self._alpha - 1))
        for key in ['firm_id', 'year']:
            simulated_df.loc[:, key] = simulated_df[key].astype(int)

        return simulated_df

    def simulate_model_old(self, n_firms, n_years):
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
            simulated_data = DataFrame(columns=['firm_id', 'year', 'capital', 'inv_rate', 'profitability', 'value'])
            simulated_data.loc[:, 'value'] = value_array
            simulated_data.loc[:, 'capital'] = self._capital_grid[capital_index]
            simulated_data.loc[:, 'profitability'] = self._profitability.state_values[profit_index]
            simulated_data.loc[:, 'profitability'] *= simulated_data.loc[:, 'capital'].apply(
                lambda x: x ** (self._alpha - 1))
            capital_policy = np.array(
                [self._capital_policy_grid[capital_index[i], profit_index[i]] for i in range(n_firms)])
            investment = self._capital_grid[capital_policy] - (1 - self._delta) * simulated_data.loc[:, 'capital']
            simulated_data.loc[:, 'inv_rate'] = investment / simulated_data.loc[:, 'capital']

            simulated_data.loc[:, 'firm_id'] = list(range(n_firms))
            simulated_data.loc[:, 'year'] = year
            simulated_data_list.append(simulated_data)

            capital_index = capital_policy.copy()
            profit_shock_series = profit_shock[:, year]

            profit_index = np.array(
                [len(current_trans[i][current_trans[i] < profit_shock_series[i]]) for i in range(n_firms)])

        simulated_data_df: DataFrame = pd.concat(simulated_data_list, ignore_index=True, sort=False)
        return simulated_data_df


class FirmValueInv(FirmValue):
    def __init__(self, alpha, delta):
        FirmValue.__init__(self, alpha, delta)
        min_i = self._delta * (2 - np.ceil(NUM_INVESTMENT / 8))
        max_i = min_i + (NUM_INVESTMENT - 1) * self._delta / 4
        self._investment_grid = np.arange(min_i, max_i, (max_i - min_i) / NUM_INVESTMENT)
        self._investment_policy = np.zeros((NUM_CAPITAL, NUM_PROFITABILITY))
        delattr(self, '_capital_policy_grid')

    def optimize(self):
        # initialize payout grid
        error_code, firm_value, all_firm_value = optimizeinv(self._alpha, self._delta, self._lambda, self._beta,
                                                             self._profitability.state_values, self._profitability.P,
                                                             self._capital_grid, self._investment_grid,
                                                             self._firm_value)
        if error_code == 0:
            self._firm_value = firm_value.copy()
            self._investment_policy = np.argmax(all_firm_value, axis=1)

        return error_code


if __name__ == '__main__':
    import time

    print(time.time())
    # for _ in range(1):
    fv = FirmValue(0.6, 0.04)
    error_code = fv.optimize()
    print(time.time())
    simulated_data = fv.simulate_model(11169, 93)

    print(time.time())
