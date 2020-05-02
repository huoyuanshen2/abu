# -*- encoding:utf-8 -*-
"""
    日内滑点卖出示例实现：开盘价卖出
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from .ABuSlippageSellBase import AbuSlippageSellBase, slippage_limit_down

__author__ = '阿布'
__weixin__ = 'abu_quant'


class AbuSlippageSellOpen(AbuSlippageSellBase):
    """示例日内滑点均价卖出类"""

    @slippage_limit_down
    def fit_price(self):
        """
        取当天交易日的开盘做为决策价格
        :return: 最终决策的当前交易卖出价格
        """

        self.sell_price = self.kl_pd_sell['open']
        return self.sell_price
