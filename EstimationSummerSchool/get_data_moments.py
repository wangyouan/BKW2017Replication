#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Filename: get_data_moments
# @Date: 2021/7/30
# @Author: Mark Wang
# @Email: markwang@connect.hku.hk

import os

import numpy as np
import pandas as pd
from pandas import DataFrame


def calculate_moments(data_df):
    mean_df = data_df.mean()
    std_df = data_df.std()

    data_moments = np.zeros(4)
    data_moments[0] = mean_df['inv_rate']
    data_moments[1] = std_df['inv_rate']
    data_moments[2] = mean_df['profitability']
    data_moments[3] = std_df['profitability']

    return data_moments


def normalize_sample_data(data_df, inv_mean, profitability_mean):
    data_df.loc[:, 'inv_rate'] = data_df['inv_rate'] - data_df['inv_rate'].mean() + inv_mean
    data_df.loc[:, 'profitability'] = data_df['profitability'] - data_df['profitability'].mean() + profitability_mean
    return data_df


if __name__ == '__main__':
    data_path = r'D:\wyatc\GoogleDrive\PhD_Life\2020_2021\StructuralEstimationinCorporateFinance\Project\Data'
    data_df: DataFrame = pd.read_csv(os.path.join(data_path, 'RealData.csv'))
    mean_df = data_df.mean()

    data_df2: DataFrame = data_df.groupby('firm_id').apply(normalize_sample_data, inv_mean=mean_df['inv_rate'],
                                                           profitability_mean=mean_df['profitability'])
    data_moments = calculate_moments(data_df2)

    if_df: DataFrame = data_df2[['firm_id', 'year']].copy()
    if_df.loc[:, 'if_mean_profit'] = data_df2['profitability'] - data_moments[2]
    if_df.loc[:, 'if_var_profit'] = if_df['if_mean_profit'].apply(lambda x: x ** 2) - data_moments[3] ** 2
    if_df.loc[:, 'if_mean_inv_rate'] = data_df2['inv_rate'] - data_moments[0]
    if_df.loc[:, 'if_var_inv_rate'] = if_df['if_mean_inv_rate'].apply(lambda x: x ** 2) - data_moments[1] ** 2

    smm_matrix = np.linalg.inv(if_df.iloc[:,-4:].cov())
