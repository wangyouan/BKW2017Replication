#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Filename: numba_method
# @Date: 2021/7/29
# @Author: Mark Wang
# @Email: markwang@connect.hku.hk

import numpy as np
import numba as nb
import pandas as pd
from pandas import DataFrame

from Utilities import inter_product
from EstimationSummerSchool import NUM_PROFITABILITY, NUM_CAPITAL, MAX_ITERATION, CONVERAGE_THRESHOLD, \
    MAX_DIFF_THRESHOLD


@nb.jit(nopython=True, parallel=False)
def optimize(alpha, delta, lambda_, beta, p_state, p_trans, capital_grid, firm_value):
    # initialize payout grid
    payout_grid = np.zeros((NUM_CAPITAL, NUM_CAPITAL, NUM_PROFITABILITY))
    for ik in range(NUM_CAPITAL):
        for ik_prime in range(NUM_CAPITAL):
            for iz in range(NUM_PROFITABILITY):
                z = p_state[iz]
                k = capital_grid[ik]
                k_prime = capital_grid[ik_prime]
                payout_grid[ik, ik_prime, iz] = z * (k ** alpha) - k_prime + (1 - delta) * k

    payout_grid = np.where(payout_grid > 0, payout_grid, (1 + lambda_) * payout_grid)
    for _ in range(MAX_ITERATION):
        firm_value_prime = firm_value @ p_trans

        all_firm_value = np.zeros((NUM_CAPITAL, NUM_CAPITAL, NUM_PROFITABILITY))
        for ik in range(NUM_CAPITAL):
            all_firm_value[ik, :, :] = payout_grid[ik, :, :] + beta * firm_value_prime

        new_firm_value = np.zeros((NUM_CAPITAL, NUM_PROFITABILITY))
        for ik in range(NUM_CAPITAL):
            for iz in range(NUM_PROFITABILITY):
                new_firm_value[ik, iz] = np.max(all_firm_value[ik, :, iz])

        model_difference = np.max(np.abs(new_firm_value - firm_value))
        if model_difference > MAX_DIFF_THRESHOLD:
            print('Cannot converage')
            return 1, firm_value, all_firm_value
        elif model_difference < CONVERAGE_THRESHOLD:
            return 0, new_firm_value, all_firm_value

        firm_value = new_firm_value.copy()

    else:
        return 2, firm_value, np.zeros((NUM_CAPITAL, NUM_CAPITAL, NUM_PROFITABILITY))


@nb.jit(nopython=True, parallel=False)
def simulate_model(delta, n_firms, n_years, firm_value, p_state, p_cdfs, capital_grid, capital_policy_grid):
    np.random.seed(1000)
    initial_state = np.random.random((n_firms, 2))
    profit_shock = np.random.random((n_firms, n_years))
    profit_index = np.array([int(i * NUM_PROFITABILITY) for i in initial_state[:, 0]], dtype=np.int64)
    capital_index = np.array([int(i * NUM_CAPITAL) for i in initial_state[:, 1]], dtype=np.int64)
    value_array = np.array([firm_value[capital_index[i], profit_index[i]] for i in range(n_firms)],
                           dtype=np.float32)

    simulated_data_list = list()
    for year in nb.prange(n_years):
        current_trans = p_cdfs[profit_index, :]

        # 'firm_id', 'year', 'capital', 'investment', 'inv_rate', 'profitability', 'value'
        simulated_data = np.zeros((n_firms, 7))
        simulated_data[:, 6] = value_array
        simulated_data[:, 2] = capital_grid[capital_index]
        simulated_data[:, 5] = p_state[profit_index]
        capital_policy = np.array(
            [capital_policy_grid[capital_index[i], profit_index[i]] for i in range(n_firms)])
        simulated_data[:, 3] = capital_grid[capital_policy] - (1 - delta) * simulated_data[:, 2]

        simulated_data[:, 0] = np.arange(n_firms)
        simulated_data[:, 1] = np.float(year)

        capital_index = capital_policy.copy()
        profit_shock_series = profit_shock[:, year]
        simulated_data_list.append(simulated_data)

        profit_index = np.array(
            [len(current_trans[i][current_trans[i] < profit_shock_series[i]]) for i in range(n_firms)])

    # simulated_data = np.vstack(simulated_data_list)
    # simulated_data[:, 5] *= np.power(simulated_data[:, 2], alpha - 1)
    # simulated_data[:, 4] = simulated_data[:, 3] / simulated_data[:, 2]
    return simulated_data_list


def simulate_model_backup(alpha, delta, n_firms, n_years, firm_value, p_state, p_cdfs, capital_grid,
                          capital_policy_grid):
    np.random.seed(1000)
    initial_state = np.random.random((n_firms, 2))
    profit_shock = np.random.random((n_firms, n_years))
    profit_index = np.array([int(i * NUM_PROFITABILITY) for i in initial_state[:, 0]], dtype=np.int64)
    capital_index = np.array([int(i * NUM_CAPITAL) for i in initial_state[:, 1]], dtype=np.int64)
    value_array = np.array([firm_value[capital_index[i], profit_index[i]] for i in range(n_firms)],
                           dtype=np.float32)

    simulated_data = np.zeros((n_firms * n_years, 7), dtype=np.float32)
    for year in range(n_years):
        current_trans = p_cdfs[profit_index, :]
        simulated_data[year * n_firms: (year + 1) * n_firms, 6] = value_array
        simulated_data[year * n_firms: (year + 1) * n_firms, 2] = capital_grid[capital_index]
        simulated_data[year * n_firms: (year + 1) * n_firms, 5] = p_state[profit_index]
        capital_policy = np.array(
            [capital_policy_grid[capital_index[i], profit_index[i]] for i in range(n_firms)])
        simulated_data[year * n_firms: (year + 1) * n_firms, 3] = \
            capital_grid[capital_policy] - (1 - delta) * simulated_data[year * n_firms: (year + 1) * n_firms, 2]

        simulated_data[year * n_firms: (year + 1) * n_firms, 0] = np.arange(n_firms)
        simulated_data[year * n_firms: (year + 1) * n_firms, 1] = np.float(year)

        capital_index = capital_policy.copy()
        profit_shock_series = profit_shock[:, year]

        profit_index = np.array(
            [len(current_trans[i][current_trans[i] < profit_shock_series[i]]) for i in range(n_firms)])

    simulated_data[:, 5] *= np.power(simulated_data[:, 2], alpha - 1)
    simulated_data[:, 4] = simulated_data[:, 3] / simulated_data[:, 2]
    return simulated_data


@nb.jit(nopython=True)
def inner_plot(capital_grid, target_k):
    up_ik = len(capital_grid[capital_grid < target_k])
    if up_ik >= NUM_CAPITAL:
        up_ik = NUM_CAPITAL - 1
        low_ik = NUM_CAPITAL - 2
    elif up_ik == 0:
        up_ik = 1
        low_ik = 0
    else:
        low_ik = up_ik - 1
    return up_ik, low_ik
