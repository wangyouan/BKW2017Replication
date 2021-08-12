#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Filename: estimate_model
# @Date: 2021/7/29
# @Author: Mark Wang
# @Email: markwang@connect.hku.hk

"""
use mean and standard deviation of investment and profitability as moments

python -m EstimationSummerSchool.estimate_model
"""

import scipy.optimize as opt
import numpy as np
from pandas import DataFrame

from EstimationSummerSchool.get_data_moments import calculate_moments
from EstimationSummerSchool.value_function_smm_school import FirmValue
from EstimationSummerSchool import NUM_SIMULATED_FIRMS, NUM_SIMULATED_YEARS, NUM_ESTIMATED_YEARS, NUM_SIMULATION


def criterion(params, *args):
    alpha, delta = params
    fv = FirmValue(delta=delta, alpha=alpha)

    data_moments, weight_matrix = args
    error_code = fv.optimize()
    if error_code != 0:
        return 1e4

    sim_moments = np.zeros_like(data_moments)
    for n_sum in range(NUM_SIMULATION):
        simulated_data: DataFrame = fv.simulate_model(n_firms=NUM_SIMULATED_FIRMS, n_years=NUM_SIMULATED_YEARS,
                                                      seed=n_sum)
        sim_moments += calculate_moments(
            simulated_data[simulated_data['year'] >= (NUM_SIMULATED_YEARS - NUM_ESTIMATED_YEARS)])

    moment_diff = sim_moments / NUM_SIMULATION - data_moments
    moments_error = moment_diff.T @ weight_matrix @ moment_diff
    print('Moments errors are:', moments_error, 'Parameters are', alpha, delta)
    return moments_error


if __name__ == '__main__':
    params_init_1 = np.array([0.8349, 0.0449])
    data_moments = np.array([0.07197835, 0.00257785, 0.13112528, 0.00748829])

    results1_1 = opt.dual_annealing(criterion, x0=params_init_1, args=(data_moments, np.eye(4)),
                                    bounds=((0.01, 0.99), (0.01, 0.25)), maxiter=100)
    print(results1_1)
