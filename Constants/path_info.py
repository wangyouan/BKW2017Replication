#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Filename: path_info
# @Date: 2022/1/17
# @Author: Mark Wang
# @Email: wangyouan@gamil.com

import os


class PathInfo(object):
    ROOT_PATH = '/home/zigan/Documents/wangyouan/'

    TEMP_PATH = os.path.join(ROOT_PATH, 'study/BKW2017Replication', 'temp')

    DATABASE_PATH = os.path.join(ROOT_PATH, 'database')
    WRDS_PATH = os.path.join(DATABASE_PATH, 'wrds')
    TFN_PATH = os.path.join(WRDS_PATH, 'tfn')
    TFN_OWNER_PATH = os.path.join(TFN_PATH, 'ownership')
    TFN_S34_PATH = os.path.join(TFN_PATH, 's34')
