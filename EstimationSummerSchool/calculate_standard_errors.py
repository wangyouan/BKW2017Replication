#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Filename: calculate_standard_errors
# @Date: 2021/8/13
# @Author: Mark Wang
# @Email: markwang@connect.hku.hk

import numpy as np
from scipy.stats import chi2
from pandas import DataFrame

from EstimationSummerSchool.get_data_moments import calculate_moments
from EstimationSummerSchool.value_function_smm_school import FirmValue
from EstimationSummerSchool import NUM_SIMULATED_FIRMS, NUM_SIMULATED_YEARS, NUM_ESTIMATED_YEARS, NUM_SIMULATION


def calculate_moments3(data_df):
    mean_df = data_df.mean()
    group_mean = data_df.groupby('firm_id')[['inv_rate', 'profitability']].mean().reset_index(drop=False)
    data_df2: DataFrame = data_df.merge(group_mean, on=['firm_id'], suffixes=['', '_mean'])
    data_df2.loc[:, 'inv_rate_demean'] = data_df2['inv_rate'] - data_df2['inv_rate_mean']
    data_df2.loc[:, 'profitability_demean'] = data_df2['profitability'] - data_df2['profitability_mean']
    var_df = data_df2.var()

    data_moments = np.zeros(3)
    data_moments[0] = mean_df['inv_rate']
    #     data_moments[1] = var_df['inv_rate_demean']
    data_moments[1] = mean_df['profitability']
    data_moments[2] = var_df['profitability_demean']

    return data_moments


def get_simulation_moments_based_on_parameter_series(parameter_array):
    alpha, delta = parameter_array
    test_fv = FirmValue(delta=delta, alpha=alpha)
    error_code = test_fv.optimize()

    sim_moments = np.zeros(3)
    if error_code == 0:
        for n_sum in range(NUM_SIMULATION):
            simulated_data: DataFrame = test_fv.simulate_model(n_firms=NUM_SIMULATED_FIRMS,
                                                               n_years=NUM_SIMULATED_YEARS,
                                                               n_sim=NUM_ESTIMATED_YEARS,
                                                               seed=n_sum)
            sim_moments += calculate_moments(simulated_data)
    return error_code, sim_moments / NUM_SIMULATION


def calculate_gradient_matrix(fv):
    current_parameters = fv.get_parameter_array()
    num_parameters = len(current_parameters)
    mom_num = 3

    delta_parameters = 1e-2 * np.abs(current_parameters)
    gradient_matrix = np.zeros((mom_num, num_parameters))

    for i in range(num_parameters):
        up_parameters = current_parameters.copy()
        low_parameters = current_parameters.copy()
        # up_moments = low_moments = np.zeros(mom_num)

        while True:
            up_parameters[i] += delta_parameters[i]
            low_parameters[i] -= delta_parameters[i]

            up_err_code, up_moments = get_simulation_moments_based_on_parameter_series(up_parameters)
            low_err_code, low_moments = get_simulation_moments_based_on_parameter_series(low_parameters)
            if low_err_code + up_err_code == 0:
                break

            delta_parameters[i] /= 2

        gradient_matrix[:, i] = up_moments - low_moments

    for i in range(mom_num):
        gradient_matrix[i, :] /= (2 * delta_parameters)

    return gradient_matrix


def get_standard_error(fv, data_moments, sample_size, weight_matrix):
    gradient_matrix = calculate_gradient_matrix(fv)
    simulation_size = NUM_ESTIMATED_YEARS * NUM_SIMULATED_FIRMS
    num_mom = len(data_moments)
    sim_moments = np.zeros_like(data_moments)
    for n_sum in range(NUM_SIMULATION):
        simulated_data: DataFrame = fv.simulate_model(n_firms=NUM_SIMULATED_FIRMS,
                                                      n_years=NUM_SIMULATED_YEARS,
                                                      n_sim=NUM_ESTIMATED_YEARS,
                                                      seed=n_sum)
        sim_moments += calculate_moments(simulated_data)
    model_diff = sim_moments / NUM_SIMULATION - data_moments
    return get_standard_error_matrix(gradient_matrix, weight_matrix, simulation_size,
                                     num_mom=num_mom, model_diff=model_diff, sample_size=sample_size)


def get_standard_error_matrix(gradient_matrix, weight_matrix, simulation_size, num_mom, model_diff, sample_size):
    cov_matrix = np.linalg.inv(weight_matrix)
    # adj_cov_matrix = (1 + 1. / simulation_size) * cov_matrix
    gwg_matrix = gradient_matrix.T @ weight_matrix @ gradient_matrix
    igwg_matrix = np.linalg.inv(gwg_matrix)
    vc = (1.0 / sample_size + 1.0 / simulation_size) * igwg_matrix
    # xxx = igwg_matrix @ gradient_matrix.T
    # covar_estimation = xxx @ weight_matrix @ adj_cov_matrix @ weight_matrix.T @ xxx.T
    standard_error = np.sqrt(np.diag(vc))

    gigwgg_matrix = gradient_matrix @ igwg_matrix @ gradient_matrix.T
    eyegg_matrix = np.eye(num_mom) - gigwgg_matrix @ weight_matrix
    vpe_matrix = eyegg_matrix @ cov_matrix @ eyegg_matrix.T
    vpe_matrix *= (1.0 / sample_size + 1.0 / simulation_size)

    chi = model_diff.T @ np.linalg.pinv(vpe_matrix) @ model_diff
    j_test = 1 - chi2.cdf(chi, 1)
    moments_error = model_diff / np.sqrt(np.diag(vpe_matrix))
    return standard_error, moments_error, j_test
