# -*- encoding:utf-8 -*-
"""
    日内滑点卖出示例实现：基于收盘价
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import time

import numpy as np

import abupy
from abupy.CoreBu.ABuStore import get_and_store_stock_detail_data
from .ABuSlippageSellBase import AbuSlippageSellBase, slippage_limit_down

__author__ = '阿布'
__weixin__ = 'abu_quant'

class AbuSlippageSellClose(AbuSlippageSellBase):

    @slippage_limit_down
    def fit_price(self):
        self.sell_price = self.kl_pd_sell.close
        return self.sell_price

