#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Filename: __init__.py
# @Date: 2021/5/25
# @Author: Mark Wang
# @Email: markwang@connect.hku.hk

from pandas import DataFrame
import numpy as np
import numba as nb
from scipy.special import erf
import pandas as pd


def generate_profitability_distribution(mu, rho, sigma, number):
    """
    ln(z') = mu + rho * ln(z) + sigma
    :param mu: mean
    :param rho: coefficient
    :param sigma: variance
    :param number: number of value needed
    :return: distribution and transition matrix
    """
    f1 = np.zeros((number, number))
    f2 = np.ones((number, number))
    m = 2

    sy = sigma / np.sqrt(1 - rho ** 2)

    z = get_range(-m * sy, m * sy, number)

    w = z[1] - z[0]
    for j in range(number):
        for i in range(1, number):
            minif = (z[i] - rho * z[j] - w / 2) / sigma
            f1[i, j] = 0.5 * (1 + erf(minif / np.sqrt(2)))
            f2[i - 1, j] = 0.5 * (1 + erf(minif / np.sqrt(2)))

    z += mu
    trans = f2 - f1
    return np.exp(z), trans


def get_range(min_val, max_val, number):
    """
    Get a range of value based on the start and end value
    :param min_val: start value
    :param max_val: end value
    :param number: number of variables needed
    :return: numpy series
    """

    step = (max_val - min_val) / (number - 1)
    result_series = np.zeros(number)
    for i in range(number):
        result_series[i] = min_val + i * step

    return result_series


@nb.jit(nopython=True, parallel=False)
def inter_product(target_val, base_series):
    """
    the inter product value would be, cfraction * series[low_index] + (1 - cfraction) * series[up_index]
    :param target_val: Target value
    :param base_series: base series
    :return: cfraction, low_index, up_index
    """
    base_length = len(base_series)
    up_index = len(base_series[base_series < target_val])
    low_index = up_index - 1
    if low_index < 0:
        return 1, 0, 1
    elif up_index >= base_length:
        return 0, base_length - 2, base_length - 1
    else:
        fraction = (base_series[up_index] - target_val) / (base_series[up_index] - base_series[low_index])

        return fraction, low_index, up_index


def get_payoff_full(profit, investment, current_debt, next_debt, gamma_, rf_, delta_, lambda__):
    payoff = profit - investment - 0.5 * gamma_ * (investment ** 2) - current_debt * (1 + rf_) \
             + next_debt * (1 - delta_ + investment)

    if payoff < 0:
        payoff *= (1 + lambda__)

    return payoff


def get_firm_value(firm_id, init_profit_index, init_debt_index, firm_value_grid, debt_grid, profit_grid,
                   debt_prime_grid, debt_policy_grid, invest_grid, invest_policy_grid, gamma_, rf_, delta_,
                   lambda__, years, trans_cdf):
    result_df = DataFrame(columns=['year', 'firm_id', 'value', 'profitability', 'debt', 'investment', 'payoff'])
    current_debt = debt_grid[init_debt_index]
    current_profit_index = init_profit_index
    for year in range(years):
        index = result_df.shape[0]
        current_profit = profit_grid[current_profit_index]
        result_df.loc[index, 'year'] = year
        result_df.loc[index, 'debt'] = current_debt
        result_df.loc[index, 'profitability'] = current_profit

        # determine next period values
        cfrac, low_index, up_index = inter_product(current_debt, debt_grid)
        trans_matrix = trans_cdf[:, current_profit_index]
        debt_prime_up = debt_prime_grid[debt_policy_grid[up_index, current_profit_index]]
        debt_prime_low = debt_prime_grid[debt_policy_grid[low_index, current_profit_index]]
        invest_prime_up = invest_grid[invest_policy_grid[up_index, current_profit_index]]
        invest_prime_low = invest_grid[invest_policy_grid[low_index, current_profit_index]]

        next_debt = cfrac * debt_prime_low + (1 - cfrac) * debt_prime_up
        result_df.loc[index, 'value'] = cfrac * firm_value_grid[low_index, current_profit_index] + (1 - cfrac) \
                                        * firm_value_grid[up_index, current_profit_index]
        result_df.loc[index, 'investment'] = cfrac * invest_prime_low + (1 - cfrac) * invest_prime_up

        result_df.loc[index, 'payoff'] = get_payoff_full(current_profit, result_df.loc[index, 'investment'],
                                                         current_debt, next_debt, gamma_, rf_, delta_,
                                                         lambda__)
        next_shock = np.random.random()
        current_profit_index = len(trans_matrix[trans_matrix < next_shock])
        current_debt = next_debt

    result_df.loc[:, 'firm_id'] = firm_id
    return result_df


def get_cluster_cov(X, group):
    """ Source BKW2018 equation 23 """
    group = pd.factorize(group)[0]
    X = np.asarray(X)
    x_group_sums = np.array([np.bincount(group, weights=X[:, col])
                             for col in range(X.shape[1])]).T
    scale = np.dot(x_group_sums.T, x_group_sums)
    nobs = X.shape[0]

    return scale / nobs


if __name__ == '__main__':
    profit, trans = generate_profitability_distribution(-2.2, 0.5, 0.086, 15)
    print(profit)

    s = ''
    for i in range(15):
        for j in range(15):
            s = '{}\t{:.4f}'.format(s, trans[i, j])

        s = '{}\n'.format(s)

    print(s)
