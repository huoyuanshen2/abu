# -*- encoding:utf-8 -*-
"""
    卖出择时示例因子：突破卖出择时因子
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from .ABuFactorSellBase import AbuFactorSellBase, AbuFactorSellXD, ESupportDirection
import time, datetime
from abupy.CoreBu.ABuStore import get_and_store_stock_detail_data, get_and_store_SHSE000001_detail_data
import numpy as np
__author__ = '阿布'
__weixin__ = 'abu_quant'


class AbuFactorSellMinMax(AbuFactorSellBase):
    """最小最大阈值策略"""

    def _init_self(self, **kwargs):
        """kwargs中必须包含: 突破参数xd 比如20，30，40天...突破"""

        # 向下突破参数 xd， 比如20，30，40天...突破
        self.is_sell_today = kwargs.pop('is_sell_today', 0)
        self.xd = kwargs.pop('xd', 5)
        self.sell_n= kwargs.pop('sell_n', 30)
        self.stopWinSell = kwargs.pop('stopWinSell', 0.1) #止盈卖出比例
        self.stopLoseSell = kwargs.pop('stopLoseSell', -0.2) #止损卖出比例
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

            self.today_ind = int(today.key)
            stockCode = today.stockCode
            date_str = time.strftime("%Y-%m-%d", time.strptime(str(today.date), "%Y%m%d"))
            kl_pd = get_and_store_stock_detail_data(stockCode, date_str)
            len_pd = np.arange(0, len(kl_pd), 5)
            for i in len_pd:
                check_pd = kl_pd.iloc[i]
                if (check_pd.close - order.buy_price) / order.buy_price >= self.stopWinSell or \
                        (check_pd.close - order.buy_price) / order.buy_price <= self.stopLoseSell:
                    self.sellPrice = check_pd['close']
                    self.sellTime = check_pd['bob']
                    self.sell_today(order, usePrice=1)
                    return
                if order.keep_days >= self.sell_n:
                    # 只要超过self.sell_n即卖出
                    self.sell_today(order) if self.is_sell_today else self.sell_tomorrow(order)
                    return


            # if temp <= 0:
            #     self.sell_today(order) if self.is_sell_today else self.sell_tomorrow(order)




        # kl_pd = self.kl_pd
        # # 拿出大盘的今天
        # benchmark_today = kl_pd[kl_pd.date == today.date]
        # if benchmark_today.empty:
        #     return False
        # end_key = int(benchmark_today.iloc[0].key)
        # start_key = end_key - 60
        # if start_key < 0:
        #     return False
        # # 使用切片切出从今天开始向前N天的数据
        # kl_pd = kl_pd[start_key:end_key + 1]

        # window_close = 5
        # kl_pd['p_change_ma'] = kl_pd.p_change.rolling(window=window_close).mean()
        # kl_pd['p_change_ma_up_rate'] = (kl_pd.p_change_ma - kl_pd.p_change_ma.shift(5))
        #
        # temp = kl_pd.iloc[-1]['p_change_ma_up_rate']




