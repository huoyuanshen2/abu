# -*- encoding:utf-8 -*-
"""
    日内开盘价买入示例实现：开盘价买入
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from .ABuSlippageBuyBase import AbuSlippageBuyBase, slippage_limit_up

__author__ = '阿布'
__weixin__ = 'abu_quant'

"""外部修改默认下跌阀值使用如：abupy.slippage.sbm.g_open_down_rate = 0.02"""
g_open_down_rate = 0.07


class AbuSlippageBuyOpen(AbuSlippageBuyBase):
    """示例日内滑点均价买入类"""

    @slippage_limit_up
    def fit_price(self):
        """
        取当天交易日的开盘价做为决策价格
        :return: 最终决策的当前交易买入价格
        """

        # TODO 基类提取作为装饰器函数，子类根据需要选择是否装饰，并且添加上根据order的call，put明确细节逻辑
        if self.kl_pd_buy.pre_close == 0 or self.kl_pd_buy.open == 0:
            # 前一日收盘价或今日开盘价为0，放弃单子
            return np.inf
        # 买入价格为当天开盘价
        self.buy_price = self.kl_pd_buy['open']
        # 返回最终的决策价格
        return self.buy_price
