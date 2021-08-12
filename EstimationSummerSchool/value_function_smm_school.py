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

from EstimationSummerSchool.numba_method import optimize, simulate_model
from EstimationSummerSchool import NUM_PROFITABILITY, NUM_CAPITAL, beta, rho, sigma, lambda_


class FirmValue(object):
    def __init__(self, alpha, delta, beta_=beta, rho_=rho, sigma_z=sigma, lambda__=lambda_):
        self._alpha = alpha
        self._delta = delta
        self._beta = beta_
        self._rho = rho_
        self._sigma = sigma_z
        self._lambda = lambda__

        self._profitability = tauchen(self._rho, self._sigma, m=2, n=NUM_PROFITABILITY)
        self._profitability.state_values = np.exp(self._profitability.state_values)
        capital_min = 0.5 * (alpha * self._profitability.state_values[0] * self._beta
                             / (1 - (1 - delta) * self._beta)) ** (1 / (1 - alpha))
        capital_max = (alpha * self._profitability.state_values[-1] * self._beta
                       / (1 - (1 - delta) * self._beta)) ** (1 / (1 - alpha))
        try:
            self._capital_grid = np.linspace(capital_min, capital_max, NUM_CAPITAL, endpoint=True, dtype=np.float32)
        except ValueError as e:
            print(self._alpha, self._beta)
            raise ValueError(e)
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

    def simulate_model(self, n_firms, n_years, seed=100):
        simulated_results = simulate_model(self._delta, n_firms, n_years, seed,
                                           self._profitability.state_values, self._profitability.cdfs,
                                           self._capital_grid, self._capital_policy_grid)
        simulated_result = np.vstack(simulated_results)

        simulated_df = DataFrame(simulated_result,
                                 columns=['firm_id', 'year', 'capital', 'investment', 'inv_rate', 'profitability'])
        simulated_df.loc[:, 'inv_rate'] = simulated_df['investment'] / simulated_df.loc[:, 'capital']
        simulated_df.loc[:, 'profitability'] *= simulated_df.loc[:, 'capital'].apply(lambda x: x ** (self._alpha - 1))
        for key in ['firm_id', 'year']:
            simulated_df.loc[:, key] = simulated_df[key].astype(int)

        return simulated_df
