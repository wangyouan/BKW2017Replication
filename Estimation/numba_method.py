#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Filename: numba_method
# @Date: 2021/8/5
# @Author: Mark Wang
# @Email: markwang@connect.hku.hk

import os

import numpy as np
import pandas as pd
from pandas import DataFrame
import numba as nb

from Utilities import inter_product


@nb.jit(nopython=True)
def optimize_model(delta, beta, firm_value, payoff, transition_matrix, debt_grid, debt_prime_grid, investment_grid,
                   int_constant_array, float_constant_array):
    p_num, z_num, p_next_num, i_num, max_iteration = int_constant_array
    coverage_threshold, diff_max_threshold = float_constant_array

    all_val = np.zeros((p_num, p_next_num, i_num, z_num))
    queue_i = np.zeros((p_next_num, i_num, z_num))

    for _ in nb.prange(max_iteration):
        expected_fv = np.dot(firm_value, transition_matrix)
        expected_fv_prime = np.zeros((p_next_num, z_num))
        for ip in nb.prange(p_next_num):
            cfrac, ipd, ipu = inter_product(debt_prime_grid[ip], debt_grid)
            expected_fv_prime[ip, :] = expected_fv[ipd, :] * cfrac + expected_fv[ipu, :] * (1 - cfrac)

        for i in range(i_num):
            queue_i[:, i, :] = expected_fv_prime * (1 - delta + investment_grid[i])

        firm_value_next = firm_value.copy()
        for ip in range(p_num):
            all_val[ip, :, :, :] = payoff[ip, :, :, :] + beta * queue_i
            for iz in range(z_num):
                firm_value_next[ip, iz] = np.max(all_val[ip, :, :, iz])

        model_diff = np.max(np.abs(firm_value_next - firm_value))
        # if _ % 100 == 0:
        #     print('Iteration: %d, Model difference: %f' % (_, model_diff))
        if model_diff < coverage_threshold:
            return 0, firm_value_next, all_val
        elif model_diff > diff_max_threshold:
            return 2, firm_value, all_val

        firm_value = firm_value_next.copy()

    else:
        return 1, firm_value, all_val


@nb.jit(nopython=True)
def get_payoff_matrix(float_constant_array, int_constant_array, profitability_grid, investment_grid, debt_grid,
                      debt_prime_grid):
    p_num, z_num, p_next_num, i_num, max_iteration = int_constant_array
    gamma, delta, rf_rate, lambda_ = float_constant_array
    payoff = np.zeros((p_num, p_next_num, i_num, z_num))

    for ip in range(p_num):
        for ip_prime in range(p_next_num):
            for ii in range(i_num):
                for iz in range(z_num):
                    profitability = profitability_grid[iz]
                    investment = investment_grid[ii]
                    debt = debt_grid[ip]
                    next_debt = debt_prime_grid[ip_prime]

                    payoff[ip, ip_prime, ii, iz] = profitability - investment - 0.5 * gamma * (investment ** 2) \
                                                   - debt * (1 + rf_rate) + next_debt * (1 - delta + investment)

    payoff = np.where(payoff > 0, payoff, (1 + lambda_) * payoff)
    return payoff


@nb.jit(nopython=True)
def simulated_model(n_firms, n_years, firm_value, debt_grid, debt_prime_grid, debt_policy_matrix, transition_matrix,
                    investment_grid, invest_policy_matrix, int_constant_array, float_constant_array, profitability_grid,
                    random_profitability_state):
    p_num, z_num, p_next_num, i_num, max_iteration = int_constant_array
    gamma, delta, rf_rate, lambda_ = float_constant_array
    profit_index = np.array([int(i * z_num) for i in random_profitability_state[:, 0]])
    debt_index = np.array([int(i * p_num) for i in random_profitability_state[:, 0]])
    value_array = np.array([firm_value[debt_index[i], profit_index[i]] for i in range(n_firms)])
    investment_array = np.array([investment_grid[invest_policy_matrix[debt_index[i], profit_index[i]]] for i in
                                 range(n_firms)])
    debt_array = np.array([debt_grid[ik] for ik in debt_index])
    trans_cdf = transition_matrix.copy()

    for i in range(z_num - 1):
        trans_cdf[i + 1, :] += trans_cdf[i, :]

    # run simulation
    simulated_data_list = list()
    for year_i in range(n_years):
        # profitability, debt, value, investment, payoff
        simulated_data = np.zeros((n_firms, 5))

        simulated_data[:, 0] = np.array([profitability_grid[ip] for ip in profit_index])
        simulated_data[:, 1] = debt_array
        simulated_data[:, 2] = value_array

        # get next period profitability
        next_shock = random_profitability_state[:, year_i + 1]
        for i in range(n_firms):
            # save current data
            profitability = simulated_data[i, 0]
            debt = simulated_data[i, 1]

            # determine next period values
            cfrac, low_index, up_index = inter_product(debt, debt_grid)
            trans_matrix = trans_cdf[:, profit_index[i]]
            current_profit_index = profit_index[i]
            debt_prime_up = debt_prime_grid[debt_policy_matrix[up_index, current_profit_index]]
            debt_prime_low = debt_prime_grid[debt_policy_matrix[low_index, current_profit_index]]
            invest_prime_up = investment_grid[invest_policy_matrix[up_index, current_profit_index]]
            invest_prime_low = investment_grid[invest_policy_matrix[low_index, current_profit_index]]

            debt_array[i] = cfrac * debt_prime_low + (1 - cfrac) * debt_prime_up
            value_array[i] = cfrac * firm_value[low_index, current_profit_index] + (1 - cfrac) \
                             * firm_value[up_index, current_profit_index]
            investment_array[i] = cfrac * invest_prime_low + (1 - cfrac) * invest_prime_up

            payoff = profitability - investment_array[i] - 0.5 * gamma * (investment_array[i] ** 2) \
                     - debt * (1 + rf_rate) + debt_array[i] * (1 - delta + investment_array[i])
            simulated_data[i, 4] = payoff if payoff > 0 else (1 + lambda_) * payoff
            profit_index[i] = len(trans_matrix[trans_matrix < next_shock[i]])

        simulated_data[:, 3] = investment_array
        simulated_data_list.append(simulated_data)
    return simulated_data_list
