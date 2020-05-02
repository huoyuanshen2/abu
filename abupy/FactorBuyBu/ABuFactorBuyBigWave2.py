# -*- encoding:utf-8 -*-
"""
    买入择时示例因子：大浪陶金2次起飞
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
class AbuBigWave2(AbuFactorBuyXD, BuyCallMixin):
    """大浪陶金2次起飞策略"""

    def _init_self(self, **kwargs):
        """
        """
        #是否启用突破因子，默认关闭。
        self.bigWave2Able = kwargs.pop('bigWave2Able', 0)
        self.bigWave2Status = 0  # 默认订单状态
        self.bigWave2XD = 3  # 一次突破后回踩次数
        self.bigWave2XDStatus = 0  # 一次突破后回踩次数当前状态
        self.bigWave = kwargs.pop('bigWave', 0.1)  # 最大跌幅

        #交易当天大盘最近N趋势角度，大于阈值算通过。
        self.benchXdAble = kwargs.pop('benchXdAble', 0)
        self.benchXd = kwargs.pop('benchXd', 5)
        self.targetAng = kwargs.pop('targetAng', 0)

        super(AbuBigWave2, self)._init_self(**kwargs)
        # 在输出生成的orders_pd中显示的名字
        self.factor_name = '{}:Days={},GTX={},Rate={}'.format(self.__class__.__name__, self.benchmark,
                                                                                    self.benchXd, self.targetAng)

    def fit_month(self, today):
        pass

    def fit_bigWave2(self, today):
        if today.key <= 60 :
            return False
        ema10 = today.ema10
        ema60 = today.ema60
        bigWave2Status = self.bigWave2Status
        if self.bigWave2Status == 0 :
            if ema10 >= ema60 :
                bigWave2Status = 1
        if self.bigWave2Status == 1 : #em10在上，开始进入
            if ema10 < ema60 :
                bigWave2Status = 2
                self.close1 = ema10
        if self.bigWave2Status == 2 : #死叉
            if ema10 < ema60 and ema10 > today.close :
                bigWave2Status = 3
            else:
                bigWave2Status = 0
        if self.bigWave2Status == 3 : #计算落差
            if today.close >= self.close1 or today.open >= self.close1 or (today.open > ema10 and today.close <= ema10 ):  # 第一次起飞突破死叉，无技术形态，从新计算
                bigWave2Status = 0
            if ema10 < ema60 and today.close > ema10 :
                bigWave2Status = 4
                self.close2 = today.close
                self.close2Date = today.name

        if self.bigWave2Status == 4 : #出现第一次突破ema10
            if ema10 < ema60 and self.close2 > today.close and self.close2 > today.open: #价格回踩
                if self.bigWave2XDStatus < self.bigWave2XD :
                    self.bigWave2XDStatus +=1
                else :
                    bigWave2Status = 5
            else:
                bigWave2Status = 0
        if self.bigWave2Status == 5 : #处于回踩状态，等待二次起飞
            if today.open > self.close2 and  today.close > self.close2 and (self.close1 -self.close2)/self.close1 >= self.bigWave : #全阳线突破
                self.bigWave2Status = 0 #状态初始化
                return True

        self.bigWave2Status =  bigWave2Status
        return False

    def fit_benchMark(self, today):
        '''
        大盘走势因子： 最近N天取角度，大于X（bench_xd）值，则发出购买信号。
        '''
        benchmark_df = self.benchmark.kl_pd
        # 拿出大盘的今天
        benchmark_today = benchmark_df[benchmark_df.date == today.date]
        if benchmark_today.empty:
            return 0
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
        """因子处理"""
        flag = True
        #5、突破因子：突破xd周期内的最大值
        if flag and self.bigWave2Able == 1:
            flag = self.fit_bigWave2(today)
        #6、大盘走势因子： 最近N天取角度，大于X（bench_xd）值，则发出购买信号。
        # if flag and self.benchXdAble == 1:
        #     flag = self.fit_benchMark(today)
        if flag :
            return self.buy_tomorrow()
        else:
            return None