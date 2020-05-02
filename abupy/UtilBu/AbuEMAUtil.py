# -*- encoding:utf-8 -*-
"""
    EMA工具类

    eg:
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from ..CoreBu import ABuEnv
__author__ = '阿布'
__weixin__ = 'abu_quant'

log_func = logging.info if ABuEnv.g_is_ipython else print

def get_EMA(cps, days):
    '''EMA函数
    '''
    emas = cps.copy()  # 创造一个和cps一样大小的集合
    for i in range(len(cps)):
        if i == 0:
            emas[i] = cps[i]
        if i > 0:
            emas[i] = ((days - 1) * emas[i - 1] + 2 * cps[i]) / (days + 1)
    return emas
