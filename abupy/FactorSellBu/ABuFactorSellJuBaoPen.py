# -*- encoding:utf-8 -*-
"""
    卖出择时示例因子：聚宝盆卖出，只管分时卖出。
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy as np
from abupy.UtilBu import ABuDateUtil
from .ABuFactorSellBase import AbuFactorSellBase, ESupportDirection
import time, datetime
from abupy.CoreBu.ABuStore import get_and_store_stock_detail_data, get_and_store_SHSE000001_detail_data

__author__ = '阿布'
__weixin__ = 'abu_quant'


class AbuFactorSellJuBaoPen(AbuFactorSellBase):
    """n日卖出策略，不管交易现在什么结果，买入后只持有N天"""

    def _init_self(self, **kwargs):
        """kwargs中可以包含: 参数sell_n：代表买入后持有的天数，默认1天"""
        self.sell_n = kwargs.pop('sell_n', 1)
        self.not_sell_n = kwargs.pop('not_sell_n', 1)
        self.is_sell_today = kwargs.pop('is_sell_today', True)
        self.sell_type_extra = '{}:sell_n={}'.format(self.__class__.__name__, self.sell_n)

    def support_direction(self):
        """因子支持两个方向"""
        return [ESupportDirection.DIRECTION_CAll.value, ESupportDirection.DIRECTION_PUT.value]

    def fit_day(self, today, orders):
        """
        :param today: 当前驱动的交易日金融时间序列数据
        :param orders: 买入择时策略中生成的订单序列
        :return:
        """
        for order in orders:
            # 将单子的持有天数进行增加
            order.keep_days += 1

            if order.keep_days <= self.not_sell_n:
                return False

            stockCode = today.stockCode
            date_str = time.strftime("%Y-%m-%d", time.strptime(str(today.date), "%Y%m%d"))
            kl_pd = get_and_store_stock_detail_data(stockCode, date_str)
            # time.sleep(0.1)

            bench_kl_pd = get_and_store_SHSE000001_detail_data(date_str)

            if (len(bench_kl_pd) - len(kl_pd)) < 0:
                print("(len(bench_kl_pd) - len(kl_pd)) < 0 ,exit.")
            if kl_pd is None or len(kl_pd) <= 200:
                return False

            bench_kl_pd = bench_kl_pd[bench_kl_pd['bob'].isin(kl_pd['bob'].tolist())]
            bench_kl_pd.index = np.arange(0, len(bench_kl_pd))

            kl_pd['date'] = kl_pd['bob'].apply(lambda x: ABuDateUtil.date_time_str_to_int(str(x)))
            kl_pd['time'] = kl_pd['bob'].apply(lambda x: ABuDateUtil.date_time_str_to_time_str(str(x)))

            kl_pd_time = kl_pd[kl_pd.time == '093000']

            kl_pd['p_change'] = (kl_pd.close - kl_pd['close'][0]) / kl_pd['close'][0]
            bench_kl_pd['p_change'] = (bench_kl_pd.close - bench_kl_pd['close'][0]) / bench_kl_pd['close'][0]
            kl_pd['p_change_update'] = (kl_pd.p_change - bench_kl_pd.p_change)

            window_volume = 60
            window_close = 60
            kl_pd['p_change_5ma'] = kl_pd.p_change.rolling(window=window_close).mean()
            kl_pd['p_change_update_5ma'] = kl_pd.p_change_update.rolling(window=window_close).mean()
            bench_kl_pd['p_change_5ma'] = bench_kl_pd.p_change.rolling(window=window_close).mean()

            kl_pd['volume_ma'] = kl_pd.volume.rolling(window=window_volume).mean()

            kl_pd['p_change_5ma_up_rate'] = (kl_pd.p_change_5ma - kl_pd.p_change_5ma.shift(5))
            kl_pd['p_change_update_5ma_up_rate'] = (kl_pd.p_change_update_5ma - kl_pd.p_change_update_5ma.shift(5))
            bench_kl_pd['p_change_5ma_up_rate'] = (bench_kl_pd.p_change_5ma - bench_kl_pd.p_change_5ma.shift(5))
            kl_pd['zero_line'] = 0

            kl_pd['volume_ma_up_rate'] = (kl_pd.volume_ma - kl_pd.volume_ma.shift(5))
            max_p_change = kl_pd['p_change_5ma_up_rate'].max()
            max_volume = kl_pd['volume_ma_up_rate'].max()

            vs_rate = max_p_change / max_volume
            kl_pd['volume_ma_up_rate'] = (kl_pd.volume_ma - kl_pd.volume_ma.shift(5)) * vs_rate

            for i in kl_pd.index:
                if i < 60:
                    continue
                start = i - 10
                if kl_pd['p_change_5ma_up_rate'][start:i].max() < 0:
                    self.sellPrice = kl_pd['close'][i]
                    self.sellTime = kl_pd['bob'][i]
                    self.sell_today(order,usePrice = 1)
                else:
                    continue
            return False

            if order.keep_days >= self.sell_n:
                # 只要超过self.sell_n即卖出
                self.sell_tomorrow(order)
