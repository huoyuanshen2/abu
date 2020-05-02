# -*- encoding:utf-8 -*-
"""
    买入择时示例因子:基金周期策略
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import time, datetime

import numpy as np
import pandas as pd
import os
import abupy
from abupy.CoreBu import ABuEnv
from abupy.CoreBu.ABuStore import get_and_store_stock_detail_data
from abupy.SimilarBu.ABuCorrcoef import corr_matrix
from abupy.UtilBu import ABuRegUtil
from abupy.UtilBu.ABuRegUtil import regress_xy_polynomial
import matplotlib.pyplot as plt
from .ABuFactorBuyBase import AbuFactorBuyXD, BuyCallMixin
from ..IndicatorBu.ABuNDMa import calc_ma_from_prices
from ..CoreBu.ABuPdHelper import pd_resample
from ..TLineBu.ABuTL import AbuTLine

__author__ = '阿布'
__weixin__ = 'abu_quant'


class AbujiJinWave(AbuFactorBuyXD, BuyCallMixin):

    def _init_self(self, **kwargs):
        self.jiJinWaveAble = kwargs.pop('jiJinWaveAble', 0)
        self.jiJinWaveStatus = 0  # 默认订单状态
        self.windowBuy = kwargs.pop('windowBuy', 10)   #买入周期
        self.windowSell = kwargs.pop('windowSell', 10)  # 卖出周期
        self.poly = kwargs.pop('poly', 50) #线性拟合系数
        self.pears_value = kwargs.pop('pears_value', 0.95) #线性拟合系数
        self.targetUpAng = kwargs.pop('targetUpAng', 100) #上升角度
        self.targetDownAng = kwargs.pop('targetDownAng', 100) #下降角度
        self.xd_kl_pd = kwargs.pop('xd_kl_pd', 30) #数据拟合所需计算数据

        self.waveMax = kwargs.pop('waveMax', 0) #波形顶部最高峰涨幅

        self.waveMin = kwargs.pop('waveMin', 0) #波形底部最低峰涨幅
        self.skip_days_value = kwargs.pop('skip_days_value', 0) #买入因子冷却间隔

        self.stopWinSellPriceRate = kwargs.pop('stopWinSellPriceRate', 0.1)  # 第一单止盈卖出价格幅度
        self.stopLoseSellPriceRate = kwargs.pop('stopLoseSellPriceRate', -0.08)  # 第一单止损卖出价格幅度
        self.buyNextPriceGrowRate = kwargs.pop('buyNextPriceGrowRate', 0.03) #第二单及第三单买入涨幅：基于第一单涨幅超过x进行买入第二单操作
        self.stopLoseSellPriceRate2 = kwargs.pop('stopLoseSellPriceRate2', -0.05) #第二单及第三单止损卖出价格幅度:第一单、第二单都基于这个值更新。

        self.havePears = False

        super(AbujiJinWave, self)._init_self(**kwargs)
        self.factor_name = '{}:windowBuy={},targetUpAng={},targetDownAng={}'.format(self.__class__.__name__, self.windowBuy,
                                                                                    self.targetUpAng, self.targetDownAng)

    def fit_month(self, today):

        # self.lock = False
        # return None
        if self.havePears == True:
            return
        self.today_ind = int(today.key)
        kl_pd = self.pre_kl_pd
        startInt = kl_pd['date'].min()
        endInt = kl_pd['date'].max()
        start = time.strftime("%Y-%m-%d", time.strptime(str(startInt), "%Y%m%d"))
        end = time.strftime("%Y-%m-%d", time.strptime(str(endInt), "%Y%m%d"))
        benchmark = abupy.AbuBenchmark(start=start, end=end, n_folds=1)
        bench_pd = benchmark.kl_pd
        kl_pd_len = len(kl_pd)
        bench_pd_len = len(bench_pd)
        if kl_pd_len != bench_pd_len:
            kl_pd['dayRateUpdated'] = kl_pd.p_change
        else:
            kl_pd['dayRateUpdated'] = kl_pd.p_change - bench_pd.p_change
        kl_pd['dayRateSum_old'] = kl_pd.dayRateUpdated.rolling(window=self.windowBuy).sum()
        kl_pd['dayRateSum_old'].fillna(value=0, inplace=True)
        kl_pd.sort_values(by='date', ascending=True, inplace=True)
        x = np.arange(0, len(kl_pd.dayRateSum_old))
        y_fit = regress_xy_polynomial(x, kl_pd.dayRateSum_old, poly=self.poly, zoom=False, show=False)
        kl_pd['y_fit'] = y_fit
        kl_pd_pears = pd.DataFrame(kl_pd, columns=['y_fit', 'dayRateSum_old'])
        from abupy.SimilarBu.ABuCorrcoef import corr_matrix
        pears_pd = corr_matrix(kl_pd_pears, similar_type=abupy.ECoreCorrType('pears'))
        pears_value = round(pears_pd.iloc[1:2, 0:1].values[0][0], 2)
        if pears_value < self.pears_value:
            self.lock = True
            self.havePears = True
            print("----------- pears is bad,stockCode={}".format(today.stockCode))
            return
        self.lock = False
        self.havePears = True
        print("************** pears is OK,stockCode={}".format(today.stockCode))

    def fit_jiJinWave(self, today):
        date = today.date
        kl_pd = self.combine_kl_pd
        kl_pd = kl_pd[kl_pd.date <= date][-30:]
        kl_pd['dayRateSum_old_5ma'] = kl_pd.p_change.rolling(window=self.windowBuy).sum()
        kl_pd['dayRateSum_old_5ma'].fillna(value=0, inplace=True)

        jiaodu_01 = (kl_pd['dayRateSum_old_5ma'][-1] - kl_pd['dayRateSum_old_5ma'][-5]) / 5
        jiaodu_02 = (kl_pd['dayRateSum_old_5ma'][-2] - kl_pd['dayRateSum_old_5ma'][-7]) / 5
        jiaodu_03 = (kl_pd['dayRateSum_old_5ma'][-3] - kl_pd['dayRateSum_old_5ma'][-8]) / 5
        if jiaodu_03 <= 0 and jiaodu_02 >= 0 and jiaodu_01 > jiaodu_02:
            return True
        else:
            return False

        kl_pd['pChangeSum'] = kl_pd.p_change.rolling(window=self.windowBuy).sum()
        kl_pd['pChangeSum'].fillna(value=0, inplace=True)
        kl_pd.sort_values(by='date', ascending=True, inplace=True)
        xdPd = kl_pd['pChangeSum'][-int(self.xd):]

        startInt = kl_pd['date'].min()
        endInt = kl_pd['date'].max()
        start = time.strftime("%Y-%m-%d", time.strptime(str(startInt), "%Y%m%d"))
        end = time.strftime("%Y-%m-%d", time.strptime(str(endInt), "%Y%m%d"))
        benchmark = abupy.AbuBenchmark(start=start, end=end, n_folds=1)
        bench_pd = benchmark.kl_pd
        lastPd = kl_pd['pChangeSum'][-6:]

        kl_pd_len = len(kl_pd)
        bench_pd_len = len(bench_pd)
        if kl_pd_len != bench_pd_len:
            kl_pd['dayRateUpdated'] = kl_pd.p_change
        else:
            kl_pd['dayRateUpdated'] = kl_pd.p_change - bench_pd.p_change

        kl_pd['dayRateSum_old'] = kl_pd.dayRateUpdated.rolling(window=self.windowBuy).sum()
        kl_pd['dayRateSum_old'].fillna(value=0, inplace=True)


        # flagBuy2 = self.cacleLatePrice(today)
        # if flagBuy2 == True:
        #     return True

        if 1 == 1:
            # aashow = 1
            if aashow == 1:
                from abupy.UtilBu.AbuJiJinDataUtil import jiJinPlotWave
                jiJinPlotWave(kl_pd=kl_pd, bench_pd=bench_pd, jiJinCodes=today.stockCode, windowBuy=self.windowBuy,
                              windowSell=self.windowSell,
                              poly=50, showJiJinOldPchangeWave=True, showWindowBuy=False, showWindowSell=False,
                              pltShow=False, index=0, ax_cnt=3)
                jiJinPlotWave(kl_pd=kl_pd, bench_pd=bench_pd, jiJinCodes=today.stockCode, windowBuy=self.windowBuy,
                              windowSell=self.windowSell,
                              poly=50, showJiJinOldPchangeWave=False, showWindowBuy=True, showWindowSell=False,
                              pltShow=False, index=1, ax_cnt=3)
                jiJinPlotWave(kl_pd=kl_pd, bench_pd=bench_pd, jiJinCodes=['000001'], windowBuy=self.windowBuy,
                              windowSell=self.windowSell,
                              poly=50, showJiJinOldPchangeWave=False, showWindowBuy=False, showBenchWave=True,
                              showWindowSell=False,
                              pltShow=False, index=2, ax_cnt=3)
                from abupy.MarketBu.ABuMarketDrawing import K_SAVE_CACHE_PNG_ROOT
                from abupy.UtilBu import ABuDateUtil
                save_dir = os.path.join(K_SAVE_CACHE_PNG_ROOT, ABuDateUtil.current_str_date())
                png_dir = os.path.join(save_dir, today.stockCode)
                from abupy.UtilBu import ABuFileUtil
                ABuFileUtil.ensure_dir(png_dir)
                r_cnt = 0
                while True:
                    png_name = '{}{}.png'.format(png_dir, '' if r_cnt == 0 else '-{}'.format(r_cnt))
                    if not ABuFileUtil.file_exist(png_name):
                        break
                    r_cnt += 1
                # png_name = '{}.png'.format(png_dir)
                plt.savefig(png_name)
                # plt.show()
            aaFlag = 1
            aaaskip_day = 0
            self.skip_days = aaaskip_day

            return True if aaFlag == 1 else False
            # return flag

        return False

    def fit_day(self, today):
        if self.lock :
            return None
        flag = True
        if flag and self.jiJinWaveAble == 1:
            flag = self.fit_jiJinWave(today)
        if flag :
            return self.buy_tomorrow()
        else:
            return None
    def cacleLatePrice(self,today):
        orders = self.orders
        #1、得到当前股票关联的keep状态订单，组成列表。
        #4、根据序列号订单价格和今天价格，判断是否更新价格。
        ordersNew = []
        stockCode = today.stockCode
        lastOrder = None
        inGroupNum = 0
        for order in orders:
            if order.sell_type != 'keep':
                continue
            if order.buy_symbol == stockCode:
                ordersNew.append(order)
                if order.inGroupNum >= inGroupNum :
                    lastOrder = order
                    inGroupNum = order.inGroupNum
        if lastOrder == None:
            return False
        order0 = None
        order1 = None
        for order in ordersNew:
            if order.inGroupNum == 0:
                order0 = order
            if order.inGroupNum == 1:
                order1 = order
            if order.inGroupNum == 2:
                order2 = order
        if inGroupNum == 0: #已有一个订单，判断是否要构建第二个订单，如果构建，需要更新之前的订单。
            if (today.close - order0.buy_price)/order0.buy_price >= self.buyNextPriceGrowRate:
                order0.stopLoseSellPriceRate = self.stopLoseSellPriceRate2
                return True
        elif inGroupNum == 1: #已有两个订单
            if (today.close - order1.buy_price)/order1.buy_price >= self.buyNextPriceGrowRate:
                order0.stopLoseSellPriceRate = self.buyNextPriceGrowRate #第一个订单提高止损卖出
                order1.stopLoseSellPriceRate = self.stopLoseSellPriceRate2
                return True
        elif inGroupNum == 3: #已有三个订单，不在买入新订单。
            if (today.close - order1.buy_price)/order1.buy_price >= self.buyNextPriceGrowRate:
                order0.stopLoseSellPriceRate = self.buyNextPriceGrowRate * 2 #第一个订单提高止损卖出
                order1.stopLoseSellPriceRate = self.buyNextPriceGrowRate
                order2.stopLoseSellPriceRate = self.stopLoseSellPriceRate2
                return False