#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Filename: constants
# @Date: 2021/5/25
# @Author: Mark Wang
# @Email: markwang@connect.hku.hk

class Constants(object):
    ###########################################
    # Constants
    ###########################################
    # risk-free rate
    RF = 0.02

    # corporate income tax
    TAU_C = 0.1
    BETA = 1 / (1 + RF * (1 + TAU_C))

    # define some grid parameters
    Z_NUM = 15
    P_NUM = 24
    P_NEXT_NUM = 64
    I_NUM = 61

    N_FIRMS = 900
    N_YEARS = 58

    # parameters for maximize value iteration
    MAX_ITERATION = 3000
    VALUE_FUNCTION_ITERATION = 50
    COVERAGE_THRESHOLD = 1e-4
    DIFF_MAX_THRESHOLD = 1e16

    # Unknown useful variables
    STR = 2
    DELP = 4
