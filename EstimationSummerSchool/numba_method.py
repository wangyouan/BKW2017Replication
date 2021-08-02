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
    MAX_DIFF_THRESHOLD, NUM_INVESTMENT, CAPITAL_MAX, CAPITAL_MIN


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


# @nb.jit(nopython=True)
def optimizeinv(alpha, delta, lambda_, beta, p_state, p_trans, capital_grid, investment_grid, firm_value):
    # initialize payout grid
    payout_grid = np.zeros((NUM_CAPITAL, NUM_INVESTMENT, NUM_PROFITABILITY))

    for ik in range(NUM_CAPITAL):
        for ii in range(NUM_INVESTMENT):
            for iz in range(NUM_PROFITABILITY):
                z = p_state[iz]
                k = capital_grid[ik]
                i_rate = investment_grid[ii]
                k_prime = k * (i_rate + 1 - delta)
                if k_prime > CAPITAL_MAX:
                    i_rate = CAPITAL_MAX / k - 1 + delta
                elif k_prime < CAPITAL_MIN:
                    i_rate = CAPITAL_MIN / k - 1 + delta
                payout_grid[ik, ii, iz] = z * (k ** alpha) - i_rate * k

    payout_grid = np.where(payout_grid > 0, payout_grid, (1 + lambda_) * payout_grid)

    # guess a invest policy
    invest_policy_grid = np.ones((NUM_CAPITAL, NUM_PROFITABILITY), dtype=np.int32)

    for _ in range(MAX_ITERATION):

        # push forward the capital policy 50 times
        new_firm_value = firm_value.copy()
        for __ in range(50):
            for ik in range(NUM_CAPITAL):
                for iz in range(NUM_PROFITABILITY):
                    ii = invest_policy_grid[ik, iz]
                    new_firm_value[ik, iz] = payout_grid[ik, ii, iz]
                    k_prime = min(max(capital_grid[ik] * (1 - delta + investment_grid[ii]), CAPITAL_MIN), CAPITAL_MAX)
                    cfrac, up_ik, low_ik = inter_product(k_prime, capital_grid)

                    for iz_prime in range(NUM_PROFITABILITY):
                        new_firm_value[ik, iz] += beta * p_trans[iz, iz_prime] * (
                                cfrac * firm_value[low_ik, iz_prime] + (1 - cfrac) * firm_value[up_ik, iz_prime])

            firm_value = new_firm_value.copy()

        new_firm_value = firm_value.copy()
        new_invest_policy = invest_policy_grid.copy()
        for ik in range(NUM_CAPITAL):
            for iz in range(NUM_PROFITABILITY):
                rhs = np.zeros(NUM_INVESTMENT)
                for ii in range(NUM_INVESTMENT):
                    rhs[ii] = payout_grid[ik, ii, iz]
                    inv_rate = investment_grid[ii]
                    k_prime = min(max(capital_grid[ik] * (1 - delta + inv_rate), CAPITAL_MIN), CAPITAL_MAX)
                    cfrac, up_ik, low_ik = inter_product(k_prime, capital_grid)
                    for iz_prime in range(NUM_PROFITABILITY):
                        rhs[ii] += beta * p_trans[iz, iz_prime] * (
                                cfrac * firm_value[low_ik, iz_prime] + (1 - cfrac) * firm_value[up_ik, iz_prime])
                new_firm_value[ik, iz] = np.max(rhs)
                new_invest_policy[ik, iz] = np.argmax(rhs)

        policy_diff = np.max(np.abs(new_invest_policy - invest_policy_grid))
        if policy_diff < 1:
            return 0, new_firm_value, new_invest_policy

        firm_value = new_firm_value.copy()
        invest_policy_grid = new_invest_policy.copy()

    else:
        return 2, firm_value, np.zeros((NUM_CAPITAL, NUM_PROFITABILITY), dtype=np.int32)


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
    for year in range(n_years):
        current_trans = p_cdfs[profit_index, :]
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


@nb.jit(nopython=True, parallel=False)
def simulate_model_invest(delta, n_firms, n_years, firm_value, p_state, p_cdfs, capital_grid, invest_grid,
                          invest_policy):
    np.random.seed(1000)
    initial_state = np.random.random((n_firms, 2))
    profit_shock = np.random.random((n_firms, n_years))
    profit_index = np.array([int(i * NUM_PROFITABILITY) for i in initial_state[:, 0]], dtype=np.int64)
    capital_index = np.array([int(i * NUM_CAPITAL) for i in initial_state[:, 1]], dtype=np.int64)
    value_array = np.array([firm_value[capital_index[i], profit_index[i]] for i in range(n_firms)],
                           dtype=np.float32)
    capital_array = capital_grid[capital_index]

    simulated_data_list = list()
    for year in range(n_years):
        simulated_data = np.zeros((n_firms, 7))
        simulated_data[:, 6] = value_array
        simulated_data[:, 5] = p_state[profit_index]
        simulated_data[:, 2] = capital_array
        invest_array = np.zeros(n_firms)

        for firm_id in range(n_firms):
            k = capital_array[firm_id]

            up_ik, low_ik = inner_plot(capital_grid, k)

            cfrac = (k - capital_grid[low_ik]) / (capital_grid[up_ik] - capital_grid[low_ik])
            inv_rate = cfrac * invest_grid[invest_policy[low_ik, profit_index[firm_id]]] + (
                    1 - cfrac) * invest_grid[invest_policy[up_ik, profit_index[firm_id]]]
            k_prime = k * (1 - delta + inv_rate)
            if k_prime > CAPITAL_MAX:
                k_prime = CAPITAL_MAX
                inv_rate = k_prime / k - 1 + delta
            elif k_prime < CAPITAL_MIN:
                k_prime = CAPITAL_MIN
                inv_rate = k_prime / k - 1 + delta

            invest_array[firm_id] = inv_rate
            capital_array[firm_id] = k_prime

        simulated_data[:, 4] = invest_array
        simulated_data[:, 0] = np.arange(n_firms)
        simulated_data[:, 1] = year

        simulated_data_list.append(simulated_data)
        profit_shock_series = profit_shock[:, year]
        current_trans = p_cdfs[profit_index, :]
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
