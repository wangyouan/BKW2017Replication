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
from EstimationSummerSchool import NUM_SIMULATED_FIRMS, NUM_SIMULATED_YEARS, NUM_ESTIMATED_YEARS


def criterion(params, *args):
    alpha, delta = params
    fv = FirmValue(delta=delta, alpha=alpha)

    data_moments, weight_matrix = args
    error_code = fv.optimize()
    if error_code != 0:
        return 1e4

    simulated_data: DataFrame = fv.simulate_model(n_firms=NUM_SIMULATED_FIRMS, n_years=NUM_SIMULATED_YEARS)
    simulated_data2: DataFrame = simulated_data[
        simulated_data['year'] >= (NUM_SIMULATED_YEARS - NUM_ESTIMATED_YEARS)].copy()
    sim_moments = calculate_moments(simulated_data2)

    moment_diff = sim_moments - data_moments
    moments_error = moment_diff.T @ weight_matrix @ moment_diff
    print('Moments errors are:', moments_error)
    return moments_error


if __name__ == '__main__':
    params_init_1 = np.array([0.8349, 0.0449])
    data_moments = [0.071978, 0.074367, 0.131125, 0.136623]

    results1_1 = opt.dual_annealing(criterion, x0=params_init_1, args=(data_moments, np.eye(4)),
                                    bounds=((0, 1), (0.01, 0.25)), maxiter=100)
    print(results1_1)
