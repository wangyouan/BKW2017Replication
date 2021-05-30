#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Filename: estimate_model
# @Date: 2021/5/29
# @Author: Mark Wang
# @Email: markwang@connect.hku.hk

import scipy.optimize as opt
import numpy as np
from pandas import DataFrame

from value_function import FirmValue


def get_moments(fv: FirmValue):
    """
    Firm
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
    sim_data_valid: DataFrame = sim_data.iloc[-8:].copy()
    result_series = np.zeros(8)
    mean_data = sim_data_valid.mean()
    var_data = sim_data_valid.var()
    result_series[0] = mean_data['investment']
    result_series[1] = var_data['investment']
    result_series[2] = mean_data['debt']
    result_series[3] = var_data['debt']
    result_series[4] = mean_data['payoff']
    result_series[5] = var_data['payoff']
    result_series[6] = mean_data['profitability']
    result_series[7] = var_data['profitability']

    return result_series


def get_moments_error(data_mom, sim_mom, weighted_matrix):
    err_mom = data_mom - sim_mom
    crit_val = err_mom.T @ weighted_matrix @ err_mom
    return crit_val


def criterion(params):
    mu, rho, sigma, delta, gamma, theta, lambda_ = params
    fv = FirmValue(delta=delta, mu=mu, rho=rho, sigma=sigma, theta=theta, lambda_=lambda_, gamma=gamma)
    error_code = fv.optimize_terry()
    if error_code != 0:
        return 1e18
    data_moments = np.array([0.0768111297195329, 0.0032904184631855, 0.1885677166674841, 0.0285271524764669,
                             0.0012114713963756, 0.0058249053810193, 0.1421154126428439, 0.0080642043112130])
    sim_moments = get_moments(fv)
    moments_error = get_moments_error(data_moments, sim_moments, np.eye(8))
    print('Moments errors are:', moments_error)
    return moments_error


if __name__ == '__main__':
    params_init_1 = np.array([-2.2067, 0.8349, 0.3594, 0.0449, 29.9661, 0.3816, 0.1829])
    results1_1 = opt.minimize(criterion, params_init_1, method='L-BFGS-B',
                              bounds=(
                                  (-6.5, -0.5), (0.3, 0.9), (0.05, 0.6), (0.01, 0.2), (3, 30), (0.1, 0.7), (0.01, 0.25)))
    print(results1_1)
