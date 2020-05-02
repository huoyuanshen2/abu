# -*- encoding:utf-8 -*-
"""
    买入择时示例因子：连续N次量价齐跌
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import time, datetime

import numpy as np
import pandas as pd

from abupy.CoreBu import ABuEnv
from abupy.CoreBu.ABuStore import get_and_store_stock_detail_data
from abupy.UtilBu import ABuRegUtil
from .ABuFactorBuyBase import AbuFactorBuyXD, BuyCallMixin
from ..IndicatorBu.ABuNDMa import calc_ma_from_prices
from ..CoreBu.ABuPdHelper import pd_resample
from ..TLineBu.ABuTL import AbuTLine

__author__ = '阿布'
__weixin__ = 'abu_quant'


# noinspection PyAttributeOutsideInit
class AbubigNDown(AbuFactorBuyXD, BuyCallMixin):
    """连续N次量价齐跌策略"""

    def _init_self(self, **kwargs):
        """
        """
        #是否启用突破因子，默认关闭。
        self.bigNDownAble = kwargs.pop('bigNDownAble', 0)
        self.bigNDownStatus = 0  # 默认订单状态
        self.downDays = kwargs.pop('downDays', 3)   #连跌次数
        self.downRate = kwargs.pop('downRate', -0.1)  # 连续跌幅
        self.openJClose = kwargs.pop('openJClose', 0.05) #实体落差距离比例

        #交易当天大盘最近N趋势角度，大于阈值算通过。
        self.benchXdAble = kwargs.pop('benchXdAble', 0)
        self.benchXd = kwargs.pop('benchXd', 5)
        self.targetAng = kwargs.pop('targetAng', 0)

        super(AbubigNDown, self)._init_self(**kwargs)
        # 在输出生成的orders_pd中显示的名字
        self.factor_name = '{}:Days={},GTX={},Rate={}'.format(self.__class__.__name__, self.benchmark,
                                                                                    self.benchXd, self.targetAng)

    def fit_month(self, today):
        pass

    def fit_bigNDown(self, today):
        bigNDownStatus = self.bigNDownStatus
        if today.p_change < self.downRate  and today.volume < self.yesterday.volume and \
                (today.open-today.close)/today.open > self.openJClose:
            if bigNDownStatus <= self.downDays :
                bigNDownStatus +=1
                if bigNDownStatus == self.downDays : #符合标准，成交。
                    self.bigNDownStatus = 0

                    return True
        else :
            bigNDownStatus = 0
        self.bigNDownStatus =  bigNDownStatus
        return False

    def fit_benchMark(self, today):
        '''
        大盘走势因子： 最近N天取角度，大于X（bench_xd）值，则发出购买信号。
        '''
        benchmark_df = self.benchmark.kl_pd
        # 拿出大盘的今天
        benchmark_today = benchmark_df[benchmark_df.date == today.date]
        if benchmark_today.empty:
            return False
        end_key = int(benchmark_today.iloc[0].key)
        start_key = end_key - self.benchXd
        if start_key < 0:
            return False
        # 使用切片切出从今天开始向前N天的数据
        benchmark_xd = benchmark_df[start_key:end_key + 1]
        ang = ABuRegUtil.calc_regress_deg(benchmark_xd.close, show=True) * 100
        if ang >= self.targetAng:
            return True
        else:
            return False


    def fit_day(self, today):
        flag = True
        if flag and self.bigNDownAble == 1:
            flag = self.fit_bigNDown(today)
        #6、大盘走势因子： 最近N天取角度，大于X（bench_xd）值，则发出购买信号。
        if flag and self.benchXdAble == 1:
            flag = self.fit_benchMark(today)
        if flag :
            return self.buy_tomorrow()
        else:
            return None