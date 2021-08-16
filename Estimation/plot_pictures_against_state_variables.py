#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Filename: plot_pictures_against_state_variables
# @Date: 2021/8/16
# @Author: Mark Wang
# @Email: markwang@connect.hku.hk

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Estimation.value_function import FirmValue

if __name__ == '__main__':
    save_path = os.path.join(r'D:\wyatc\Documents\Projects\CarbonTax\pictures', '20210816_pictures')
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    mu, rho, sigma, delta, gamma, theta, lambda_ = -2.2067, 0.8349, 0.3594, 0.0449, 29.9661, 0.3816, 0.1829
    fv = FirmValue(delta=delta, rho=rho, mu=mu, gamma=gamma, theta=theta, sigma=sigma, lambda_=lambda_)
    error_code = fv.optimize()

    profitability_grid = fv.get_profitability_grid()
    debt_grid = fv.get_debt_grid()
    debt_policy = fv.get_debt_policy()
    invest_policy = fv.get_invest_policy()
    firm_value_matrix = fv.get_firm_value_matrix()

    for target in ['firm_value', 'debt_policy', 'invest_policy']:
        if target == 'firm_value':
            target_grid = firm_value_matrix.copy()
            z_label = 'Firm Value'
        elif target == 'debt_policy':
            target_grid = debt_policy.copy()
            z_label = 'Debt Policy'
        else:
            target_grid = invest_policy.copy()
            z_label = 'Investment Policy'

        fig, ax = plt.subplots()
        plt.plot(debt_grid, target_grid[:, 1], 'b-', label='Low Profitability')
        plt.plot(debt_grid, target_grid[:, fv.Z_NUM // 2], 'r-', label='Middle Profitability')
        plt.plot(debt_grid, target_grid[:, fv.Z_NUM - 1], 'g-', label='High Profitability')
        ax.set_xlabel('Debt')
        ax.set_ylabel(z_label)
        ax.set_title('{} against Debt level'.format(z_label))
        ax.legend()
        fig.savefig(os.path.join(save_path, '2d_{}_debt.png'.format(target)))
        plt.clf()

        fig, ax = plt.subplots()
        plt.plot(profitability_grid, target_grid[0, :], 'b-', label='Low Profitability')
        plt.plot(profitability_grid, target_grid[fv.P_NUM // 2, :], 'r-', label='Middle Profitability')
        plt.plot(profitability_grid, target_grid[fv.P_NUM - 1, :], 'g-', label='High Profitability')
        ax.set_xlabel('Profitability')
        ax.set_ylabel(z_label)
        ax.set_title('{} against Profitability'.format(z_label))
        ax.legend()
        fig.savefig(os.path.join(save_path, '2d_{}_profitability.png'.format(target)))
        plt.clf()

        fig = plt.figure()
        ax = plt.axes(projection="3d")
        X, Y = np.meshgrid(profitability_grid, debt_grid)
        ax.plot_surface(X, Y, target_grid, rstride=1, cstride=1,
                        cmap='winter', edgecolor='none')
        ax.set_ylabel('Debt')
        ax.set_xlabel('Profitability')
        ax.set_zlabel(z_label)
        ax.set_title('{} against Debt and Profitability'.format(z_label))
        fig.savefig(os.path.join(save_path, '3d_{}_profitability_debt.png'.format(target)))
        plt.clf()
