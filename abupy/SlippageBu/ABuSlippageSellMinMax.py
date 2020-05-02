# -*- encoding:utf-8 -*-
"""
    日内滑点卖出示例实现：最小最大阈值卖出
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import time

import numpy as np

from abupy.CoreBu.ABuStore import get_and_store_stock_detail_data
from .ABuSlippageSellBase import AbuSlippageSellBase, slippage_limit_down

__author__ = '阿布'
__weixin__ = 'abu_quant'

g_max_rate = 0.05 #获利卖出阈值
g_min_rate = 0.08 #止损卖出阈值
class AbuSlippageSellMinMax(AbuSlippageSellBase):
    """示例日内滑点均价卖出类"""

    @slippage_limit_down
    def fit_price(self):
        """
        """
        stockCode = self.kl_pd_sell.stockCode
        timeArray = time.strptime(str(self.kl_pd_sell.date), "%Y%m%d")
        date_str = time.strftime("%Y-%m-%d", timeArray)

        detail_pd = get_and_store_stock_detail_data(stockCode, date_str)
        max_price = self.order.buy_price * (1+g_max_rate)
        min_price = self.order.buy_price * (1-g_min_rate)
        if detail_pd is None:
            return -np.inf
        max_price_times = detail_pd[detail_pd['price'] >= max_price]
        min_price_times = detail_pd[detail_pd['price'] <= min_price]
        first_max = max_price_times[0:1].time if len(max_price_times) > 0 else None
        first_min = min_price_times[0:1].time if len(min_price_times) > 0 else None
        if first_min is not None and first_max is not None:
            sell_price = min_price if (first_max > first_min) else max_price
        elif (first_max is None and first_min is not None) :
            sell_price = min_price
        elif (first_min is None and first_max is not None) :
            sell_price = max_price
        else :
            return -np.inf
        self.sell_price = sell_price
        return self.sell_price
