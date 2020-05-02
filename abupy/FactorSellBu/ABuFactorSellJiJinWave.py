# -*- encoding:utf-8 -*-
"""
    卖出择时示例因子：基金周期波动因子
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import time

import abupy
from .ABuFactorSellBase import AbuFactorSellBase, AbuFactorSellXD, ESupportDirection

__author__ = '阿布'
__weixin__ = 'abu_quant'


class AbuFactorSellJiJinWave(AbuFactorSellBase):
    """最小最大阈值策略"""

    def _init_self(self, **kwargs):
        """kwargs中必须包含: 突破参数xd 比如20，30，40天...突破"""

        # 向下突破参数 xd， 比如20，30，40天...突破
        self.xd = kwargs.pop('xd', 5)
        self.windowBuy = kwargs.pop('windowBuy', 5)
        # 在输出生成的orders_pd中显示的名字
        self.sell_type_extra = '{}:{}'.format(self.__class__.__name__, self.xd)

    def support_direction(self):
        """因子支持两个方向"""
        return [ESupportDirection.DIRECTION_CAll.value, ESupportDirection.DIRECTION_PUT.value]

    def fit_day(self, today, orders):
        """
        """
        for order in orders:
            if order.sell_type != 'keep':
                continue
            # 1、获取大盘数据，得到大盘当天价格
            # 2、与订单中存有的大盘记录计算，得到大盘的波动比率。
            # 3、(当前订单波动 - 大盘波动) > 计划阈值，卖出。（包括止损）
            kl_pd = self.kl_pd
            date = today.date
            kl_pd = kl_pd[kl_pd.date <= date][-30:]

            kl_pd['dayRateSum_old_5ma'] = kl_pd.p_change.rolling(window=self.windowBuy).sum()
            kl_pd['dayRateSum_old_5ma'].fillna(value=0, inplace=True)

            jiaodu_01 = (kl_pd['dayRateSum_old_5ma'][-1] - kl_pd['dayRateSum_old_5ma'][-5]) / 5
            jiaodu_02 = (kl_pd['dayRateSum_old_5ma'][-2] - kl_pd['dayRateSum_old_5ma'][-7]) / 5
            jiaodu_03 = (kl_pd['dayRateSum_old_5ma'][-3] - kl_pd['dayRateSum_old_5ma'][-8]) / 5
            if jiaodu_03 >= 0 and jiaodu_02 <= 0 and jiaodu_01 <= 0:
                self.sell_tomorrow(order)

            # kl_pd_sell = self.kl_pd.iloc[self.today_ind]



            # stockCode = today.stockCode
            # timeArray = time.strptime(str(kl_pd_sell.date), "%Y%m%d")
            # start_str = time.strftime("%Y-%m-%d", timeArray)
            # bench_pd = abupy.AbuBenchmark(start=start_str, end=start_str, n_folds=1).kl_pd
            # sell_bench_price = bench_pd['close'].values[0]
            # buy_bench_price = order.buy_bench_price
            # bench_atr = (sell_bench_price - buy_bench_price) / buy_bench_price
            # stock_atr = (order.buy_price - kl_pd_sell.close) / order.buy_price
            # dif_atr = stock_atr - bench_atr
            # if dif_atr >= order.stopWinSellPriceRate: #止盈卖出
            #     self.sell_tomorrow(order)
            # if dif_atr <= order.stopLoseSellPriceRate: #止损卖出
            #     self.sell_tomorrow(order)
