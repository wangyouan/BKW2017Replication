#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Filename: estimate_model
# @Date: 2021/5/29
# @Author: Mark Wang
# @Email: markwang@connect.hku.hk


"""
     fun: 0.00043080615655367713
 message: ['Maximum number of iteration reached']
    nfev: 3097
    nhev: 0
     nit: 100
    njev: 212
  status: 0
 success: True
       x: array([-2.25597919,  0.84899213,  0.36075207,  0.0470482 , 29.7260005 , 0.19591638,  0.03514085])
"""

import numpy as np
import pandas as pd
import scipy.optimize as opt
from pandas import DataFrame

from Estimation.value_function import FirmValue
from EstimationSummerSchool.calculate_standard_errors import get_standard_error_matrix

data_moments = np.array([0.1885677166674841, 0.0285271524764669, 0.0768111297195329, 0.0032904184631855,
                         0.0012114713963756, 0.0058249053810193, 0.1421154126428439, 0.0080642043112130])
weight_matrix = np.array([[1.10970812e+00, -5.45373300e-03, 5.73558990e-02, 4.00641400e-03, -1.10963740e-02,
                           -7.19755100e-03, 5.64500000e-03, -9.21241100e-03],
                          [-5.45373300e-03, 1.72861350e-02, -4.43486000e-04, 2.97896000e-04, -2.85893600e-03,
                           1.05785900e-03, -3.14452800e-03, 1.29062300e-03],
                          [5.73558990e-02, -4.43486000e-04, 5.08293810e-02, 3.39974400e-03, -5.32097800e-03,
                           6.54373000e-04, 2.51933490e-02, 6.81779000e-04],
                          [4.00641400e-03, 2.97896000e-04, 3.39974400e-03, 4.21314000e-04, -8.45842000e-04,
                           1.28310000e-04, 7.18566000e-04, 1.48138000e-04],
                          [-1.10963740e-02, -2.85893600e-03, -5.32097800e-03, -8.45842000e-04, 3.38320220e-02,
                           -3.81671500e-03, 3.74180570e-02, -1.36726600e-03],
                          [-7.19755100e-03, 1.05785900e-03, 6.54373000e-04, 1.28310000e-04, -3.81671500e-03,
                           1.23323600e-03, -2.44482600e-03, 3.97526000e-04],
                          [5.64500000e-03, -3.14452800e-03, 2.51933490e-02, 7.18566000e-04, 3.74180570e-02,
                           -2.44482600e-03, 1.21059595e-01, -3.28985000e-04],
                          [-9.21241100e-03, 1.29062300e-03, 6.81779000e-04, 1.48138000e-04, -1.36726600e-03,
                           3.97526000e-04, -3.28985000e-04, 1.28965600e-03]])


def get_moments(fv: FirmValue):
    """
    Simulate model and return moments
    :param fv: an optimized firm value
    :return: a list [mean investment, var investment, mean leverage, var leverage, mean payoff, var payoff,
                     mean profit, var profit]
        0.0768111297195329
        0.0032904184631855
        0.1885677166674841
        0.0285271524764669
        0.0012114713963756
        0.0058249053810193
        0.1421154126428439
        0.0080642043112130

    """
    sim_data: DataFrame = fv.simulate_model(years=FirmValue.N_YEARS, firms=FirmValue.N_FIRMS)
    sim_data_valid: DataFrame = sim_data.loc[sim_data['year'] >= FirmValue.N_SIMULATION].copy()
    result_series = np.zeros(8)
    mean_data = sim_data_valid.mean()
    var_data = sim_data_valid.var()
    result_series[2] = mean_data['investment']
    result_series[3] = var_data['investment']
    result_series[0] = mean_data['debt']
    result_series[1] = var_data['debt']
    result_series[4] = mean_data['payoff']
    result_series[5] = var_data['payoff']
    result_series[6] = mean_data['profitability']
    result_series[7] = var_data['profitability']

    return result_series


def get_moments_error(data_mom, sim_mom, weighted_matrix):
    err_mom = data_mom - sim_mom
    crit_val = err_mom.T @ weighted_matrix @ err_mom
    return crit_val


def criterion(params, *args):
    mu, rho, sigma, delta, gamma, theta, lambda_ = params
    fv = FirmValue(delta=delta, mu=mu, rho=rho, sigma=sigma, theta=theta, lambda_=lambda_, gamma=gamma)
    error_code = fv.optimize()
    if error_code != 0:
        return 10

    sim_moments = get_moments(fv)
    moments_error = get_moments_error(data_moments, sim_moments, weight_matrix)
    print('Moments errors are:', moments_error)
    return moments_error


def get_simulation_moments_based_on_parameter_series(parameter_array):
    mu, rho, sigma, delta, gamma, theta, lambda_ = parameter_array
    test_fv = FirmValue(delta=delta, mu=mu, rho=rho, sigma=sigma, theta=theta, lambda_=lambda_, gamma=gamma)
    error_code = test_fv.optimize()

    if error_code == 0:
        sim_moments = get_moments(test_fv)

    else:
        sim_moments = np.zeros(8)

    return error_code, sim_moments


def calculate_gradient_matrix(fv):
    current_parameters = fv.get_parameter_array()
    num_parameters = len(current_parameters)
    mom_num = 8

    delta_parameters = 1e-8 * np.abs(current_parameters)
    gradient_matrix = np.zeros((mom_num, num_parameters))

    for i in range(num_parameters):
        up_parameters = current_parameters.copy()
        low_parameters = current_parameters.copy()
        # up_moments = low_moments = np.zeros(mom_num)

        while True:
            up_parameters[i] += delta_parameters[i]
            low_parameters[i] -= delta_parameters[i]

            up_err_code, up_moments = get_simulation_moments_based_on_parameter_series(up_parameters)
            if up_err_code != 0:
                delta_parameters[i] /= 2
                continue

            low_err_code, low_moments = get_simulation_moments_based_on_parameter_series(low_parameters)
            if low_err_code == 0:
                break

            delta_parameters[i] /= 2

        gradient_matrix[:, i] = up_moments - low_moments

    for i in range(mom_num):
        gradient_matrix[i, :] /= (2 * delta_parameters)

    return gradient_matrix


def get_standard_error(fv, sample_size):
    gradient_matrix = calculate_gradient_matrix(fv)
    simulation_size = FirmValue.N_FIRMS * (FirmValue.N_YEARS - FirmValue.N_SIMULATION)
    num_mom = len(data_moments)

    model_diff = get_moments(fv) - data_moments
    return get_standard_error_matrix(gradient_matrix, sample_size, simulation_size, num_mom=num_mom,
                                     model_diff=model_diff)


if __name__ == '__main__':
    params_init_1 = np.array([-2.2067, 0.8349, 0.3594, 0.0449, 29.9661, 0.3816, 0.1829])
    # results1_1 = opt.minimize(criterion, params_init_1, args=(2,), method='L-BFGS-B',
    #                           options={'eps': 1e-1, 'gtol': 1e-3},
    #                           bounds=(
    #                               (-6.5, -0.5), (0.3, 0.9), (0.05, 0.6), (0.01, 0.2), (3, 30), (0.1, 0.7),
    #                               (0.01, 0.25)))
    results1_1 = opt.dual_annealing(criterion, x0=params_init_1, maxiter=100,
                                    bounds=(
                                        (-3.5, -1.5), (0.8, 0.9), (0.3, 0.4), (0.01, 0.05), (25, 30), (0.1, 0.4),
                                        (0.01, 0.25)))
    print(results1_1)
    # mu, rho, sigma, delta, gamma, theta, lambda_ = params_init_1
    # process_num: int = 2
    # fv = FirmValue(delta=delta, mu=mu, rho=rho, sigma=sigma, theta=theta, lambda_=lambda_, gamma=gamma)
    # error_code = fv.optimize_terry()
    # data_moments = np.array([0.0768111297195329, 0.0032904184631855, 0.1885677166674841, 0.0285271524764669,
    #                          0.0012114713963756, 0.0058249053810193, 0.1421154126428439, 0.0080642043112130])
    # sim_moments = get_moments(fv, process_num)
