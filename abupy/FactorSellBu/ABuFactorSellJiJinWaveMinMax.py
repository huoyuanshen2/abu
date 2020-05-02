# -*- encoding:utf-8 -*-
"""
    卖出择时示例因子：根据订单中的预设止盈、止损阈值进行卖出。
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time

import abupy
from .ABuFactorSellBase import AbuFactorSellBase, AbuFactorSellXD, ESupportDirection

__author__ = '阿布'
__weixin__ = 'abu_quant'


class AbuFactorSellJiJinWaveMinMax(AbuFactorSellBase):
    """最小最大阈值策略"""

    def _init_self(self, **kwargs):
        """kwargs中必须包含: 突破参数xd 比如20，30，40天...突破"""

        # 向下突破参数 xd， 比如20，30，40天...突破
        self.xd = kwargs.pop('xd', 5)
        self.sell_n = kwargs.pop('sell_n', 5)
        # 在输出生成的orders_pd中显示的名字
        self.sell_type_extra = '{}:{}'.format(self.__class__.__name__, self.xd)

    def support_direction(self):
        """因子支持两个方向"""
        return [ESupportDirection.DIRECTION_CAll.value, ESupportDirection.DIRECTION_PUT.value]

    def fit_day(self, today, orders):
        for order in orders:
            if order.sell_type != 'keep':
                continue
            order.keep_days += 1
            if (today.close - order.buy_price)/order.buy_price >= order.stopWinSellPriceRate or \
                    (today.close - order.buy_price)/order.buy_price <= order.stopLoseSellPriceRate \
                    or order.keep_days >= self.sell_n:
                self.sell_tomorrow(order)
