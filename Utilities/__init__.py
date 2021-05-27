#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Filename: __init__.py
# @Date: 2021/5/25
# @Author: Mark Wang
# @Email: markwang@connect.hku.hk

import numpy as np
from scipy.special import erf


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
    z = np.zeros(number)
    m = 2

    s2 = sigma ** 2
    sy = np.sqrt(s2 / (1 - rho ** 2))
    for i in range(number):
        z[i] = -m * sy + ((2 * m * sy) / (number - 1) * i)

    w = z[2] - z[1]
    for j in range(number):
        for i in range(1, number):
            minif = (z[i] - rho * z[j] - w / 2) / sigma
            f1[i, j] = 0.5 * (1 + erf(minif / np.sqrt(2)))
            f2[i - 1, j] = 0.5 * (1 + erf(minif / np.sqrt(2)))

    z += mu
    f1 = f1.T
    f2 = f2.T
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

    step = (max_val - min_val) / number
    result_series = np.zeros(number)
    for i in range(number):
        result_series[i] = min_val + i * step

    return result_series


if __name__ == '__main__':
    profit, trans = generate_profitability_distribution(-2.2, 0.5, 0.086, 15)
    print(profit)

    s = ''
    for i in range(15):
        for j in range(15):
            s = '{}\t{:.4f}'.format(s, trans[i, j])

        s = '{}\n'.format(s)

    print(s)
