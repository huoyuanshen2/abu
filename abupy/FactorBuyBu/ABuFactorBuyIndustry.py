# -*- encoding:utf-8 -*-
"""
    买入择时示例因子：基于行业涨幅比例、涨幅占比策略
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import time, datetime

import numpy as np
import pandas as pd

from ..MarketBu.ABuDataCache import load_kline_df
from ..MarketBu.ABuSymbol import code_to_symbol
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
class AbuIndustryRateBuy(AbuFactorBuyXD, BuyCallMixin):
    """示例买入行业比率策略"""

    def _init_self(self, **kwargs):
        """
            kwargs中可选参数：changeRate: 一个行业有targetCountRate比例股票涨幅高过该值，才算一个行业启动。5表示一个行业中已知股票涨幅达5%及以上才算一个有效涨幅股票。
            kwargs中可选参数：targetCountRate: 一个行业中，有多少比例符合涨幅要求算通过。30表示一个行业30%的股票涨幅达到changeRate才算行业启动。
        """
        # 单支股票涨幅比例
        # 当前股票最大涨幅比例，当日涨幅达到超过这个阈值，才考虑买入
        self.maxChangeRateAble = kwargs.pop('maxChangeRateAble', 0)
        self.maxChangeRate = kwargs.pop('maxChangeRate', 8)

        # 一个行业中，相关性高的股票中，选几只算作参考。
        self.targetCountRateAble = kwargs.pop('targetCountRateAble', 0)
        self.targetCountRate = kwargs.pop('targetCountRate', 30)
        self.changeRate = kwargs.pop('changeRate', 5)

        # 一个行业中，有多少百分比例符合涨幅要求算通过。
        self.todayGT0DataAble = kwargs.pop('todayGT0DataAble', 0)
        self.targetGT0CountRate = kwargs.pop('targetGT0CountRate', 30)

        #一天的数据中，有多少百分比例时间为最高价。
        self.maxPriceRateAble = kwargs.pop('maxPriceRateAble', 0)
        self.maxPriceRate = kwargs.pop('maxPriceRate', 50)

        #是否启用突破因子，默认开启。
        self.xdAble = kwargs.pop('xdAble', 0)

        #交易当天大盘最近N趋势角度，大于阈值算通过。
        self.benchXdAble = kwargs.pop('benchXdAble', 0)
        self.benchXd = kwargs.pop('benchXd', 5)
        self.targetAng = kwargs.pop('targetAng', 0)

        # 连续涨停因子： 在本天前，已连续涨停N次(不算本次),则发出购买信号。
        self.tradingDaysAble = kwargs.pop('tradingDaysAble', 0)
        self.tradingDays = kwargs.pop('tradingDays', 0)
        self.targetGTX = kwargs.pop('targetGTX', 0) #涨幅个数
        self.targetGTXRate = kwargs.pop('targetGTXRate', 0) #涨幅比例

        super(AbuIndustryRateBuy, self)._init_self(**kwargs)
        # 在输出生成的orders_pd中显示的名字
        self.factor_name = '{}:Days={},GTX={},Rate={}'.format(self.__class__.__name__, self.tradingDays,
                                                                                    self.targetGTX, self.targetGTXRate)

    def fit_month(self, today):
        pass

    def fit_maxChangeRate(self, today):
        '''
        涨幅因子：涨幅比例突破X值
        :param today:
        :return:
        '''
        flag = True
        if today.p_change < self.maxChangeRate:
            flag = False
        else :
            colTemp = self.fit_tradingDaysReact(today)
            if (colTemp is not None and flag and len(colTemp) >=  self.targetGTX) == False:
                flag = False
        return flag

    def fit_todayGT0Data(self, today):
        '''
        行业因子：行业内有N支股票上涨，涨幅大于X值
        :param today:
        :return:
        '''
        stockCode = today.stockCode
        stockIndustryNames = ABuEnv.industry_data['c_name'][ABuEnv.industry_data.code == int(stockCode)]
        for _, stockIndustryName in stockIndustryNames.items():
            industryData = ABuEnv.industryPchangeAllDataDic[stockIndustryName]
            industryTodayData = industryData[industryData.date == today.date][0:1]
            industryTodayData2 = industryTodayData.drop('date', axis=1, inplace=False)
            todayData = pd.DataFrame(industryTodayData2.values.T, index=industryTodayData2.columns,
                                     columns=['p_change'])
            todayGT0Data = todayData[todayData.p_change > 0]
            allCount = todayData.shape[0]
            ableCount = todayGT0Data.shape[0]
            if allCount > 0:
                countRate = (ableCount * 100) / allCount
                if countRate >= self.targetGT0CountRate:
                    return True
                else :
                    return False
    def fit_targetCountRate(self, today):
        '''
        相似度因子：行业内相似度最高的N支股票，涨幅大于X值
        :param today:
        :return:
        '''
        stockCode = today.stockCode
        stockIndustryNames = ABuEnv.industry_data['c_name'][ABuEnv.industry_data.code == int(stockCode)]
        for _, stockIndustryName in stockIndustryNames.items():
            industryPearsonrData = ABuEnv.industryPearsonrDataDic[stockIndustryName]
            industryPearsonr4One = industryPearsonrData[stockCode]
            industryPearsonr4OneOrderLimit = industryPearsonr4One.sort_values(ascending=False)[
                                             1:self.targetCountRate + 1]  # 取相关度最大的targetCountRate只
            if industryPearsonr4OneOrderLimit.size < self.targetCountRate:
                continue
            industryData = ABuEnv.industryPchangeAllDataDic[stockIndustryName]
            industryTodayData = industryData[industryData.date == today.date][0:1]
            for stockCodeTemp in industryPearsonr4OneOrderLimit.index:
                if industryTodayData.shape[0] == 0:
                    break
                stockPChange = industryTodayData[stockCodeTemp]
                stockPChange = stockPChange.values[0]
                if stockPChange is None:
                    break
                if stockPChange < self.changeRate:
                    break
                return True
        return False

    def fit_maxPriceRate(self, today):
        '''
        最高价占比因子： 当天价格数高于最大值的百分比
        '''
        stockCode = today.stockCode
        timeArray = time.strptime(str(today.date), "%Y%m%d")
        date_str = time.strftime("%Y-%m-%d", timeArray)

        stock_detail_data = get_and_store_stock_detail_data(stockCode, date_str)
        if stock_detail_data is None:
            return False
        today_max_data = stock_detail_data[stock_detail_data.price == today.close]
        con_all = stock_detail_data.shape[0]
        con_max = today_max_data.shape[0]

        maxRate = (con_max * 100) / con_all
        if maxRate  >= self.maxPriceRate:
            return True
        else :
            return False
    def fit_xd(self, today):
        '''
        最高价占比因子： 当天价格数高于最大值的百分比
        '''
        if today.close >= self.xd_kl.close.max():  # 当前价格为突破价格
            return True
        else:
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

    def fit_tradingDays(self, today):
        '''
        # 7、连续涨停因子： 在本天前，已连续涨停N次(不算本次),则发出购买信号。
        '''
        kl_react_pd = self.kl_pd
        kl_pd_key = kl_react_pd[kl_react_pd.date == today.date]['key'].values[0]

        flag = True
        columns = []

        target_data2 = kl_react_pd.iloc[kl_pd_key - (int(self.tradingDays)+1)]
        if target_data2 is not None and target_data2.p_change > 9.6 : #排除非第一次涨停
            return False
        for i in range(1,int(self.tradingDays)+1):
            target_data = kl_react_pd.iloc[kl_pd_key - i]
            if target_data is not None and target_data.p_change > 9.6 :
                if len(columns) == 0:
                    columns = self.fit_tradingDaysReact(target_data)
                else:
                    colTemp = self.fit_tradingDaysReact(target_data)
                    if colTemp is not None:
                        a = set(columns)
                        b = set(colTemp)
                        columns = a & b
                if flag and len(columns) < self.targetGTX:
                    return False
            else :
                return False
        #         判断参考标的第一天打板前一天非涨停
        for react_stock in columns :
            sh_stock_react_code = code_to_symbol(react_stock).value
            kl_react_pd, df_req_start, df_req_end = load_kline_df(sh_stock_react_code)
            kl_pd_key = kl_react_pd[kl_react_pd.date == today.date]['key'].values[0]
            target_data2 = kl_react_pd.iloc[kl_pd_key - (int(self.tradingDays) + 1)]
            if target_data2 is not None and target_data2.p_change > 9.6:
                return False

        columns = np.append(columns, today.stockCode)
        rate_dic = {}
        for react_stock in columns:
            sh_stock_react_code = code_to_symbol(react_stock).value
            # df, df_req_start, df_req_end = load_kline_df(sh_stock_react_code)
            kl_react_pd, df_req_start, df_req_end = load_kline_df(sh_stock_react_code)
            kl_pd_key = kl_react_pd[kl_react_pd.date == today.date]['key'].values[0]
            target_data2 = kl_react_pd.iloc[kl_pd_key - (int(self.tradingDays) )]
            # 判断参考标的第一天打板前一天，分时图涨停占比
            timeArray = time.strptime(str(int(target_data2.date)), "%Y%m%d")
            date_str = time.strftime("%Y-%m-%d", timeArray)

            stock_detail_data = get_and_store_stock_detail_data(react_stock, date_str)
            if stock_detail_data is None:
                flag = False
                break
            today_max_data = stock_detail_data[stock_detail_data.price == target_data2.close]
            con_all = stock_detail_data.shape[0]
            con_max = today_max_data.shape[0]
            maxRate = (con_max * 100) / con_all
            rate_dic[str(react_stock)] = maxRate
        maxRateStock = max(rate_dic, key=rate_dic.get)

        if (maxRateStock != str(today.stockCode)) or (max(rate_dic.values()) == 100): #非一字板
            flag = False

        return flag


    def fit_tradingDaysReact(self, targetDay):
        '''
        # 8、连续涨停因子的子因子，： 在本天前，已连续涨停N次(不算本次),则发出购买信号。
        '''
        stockCode = targetDay.stockCode
        stockIndustryNames = ABuEnv.industry_data['c_name'][ABuEnv.industry_data.code == int(stockCode)]
        for _, stockIndustryName in stockIndustryNames.items():
            industryData = ABuEnv.industryPchangeAllDataDic[stockIndustryName]
            industryTodayData = industryData[industryData.date == targetDay.date][0:1]
            industryTodayData2 = industryTodayData.drop('date', axis=1, inplace=False)
            todayData = pd.DataFrame(industryTodayData2.values.T, index=industryTodayData2.columns,
                                     columns=['p_change'])
            todayGTXData = todayData[todayData.p_change > self.targetGTXRate]
            return todayGTXData.index.values
        return None



    def fit_day(self, today):
        """因子处理"""
        flag = True
        # 1、涨幅因子：涨幅比例突破X值
        if flag and self.maxChangeRateAble == 1:
            flag = self.fit_maxChangeRate(today)
        # 2、行业因子：行业内有N支股票上涨，涨幅大于X值
        if flag and self.todayGT0DataAble == 1:
            flag = self.fit_todayGT0Data(today)
        # 3、相似度因子：行业内相似度最高的N支股票，涨幅大于X值
        if flag and self.targetCountRateAble == 1:
            flag = self.fit_targetCountRate(today)
        # 4、最高价占比因子： 当天价格数高于最大值的百分比
        if flag and self.maxPriceRateAble == 1:
            flag = self.fit_maxPriceRate(today)
        #5、突破因子：突破xd周期内的最大值
        if flag and self.xdAble == 1:
            flag = self.fit_xd(today)
        #6、大盘走势因子： 最近N天取角度，大于X（bench_xd）值，则发出购买信号。
        if flag and self.benchXdAble == 1:
            flag = self.fit_benchMark(today)
        # 7、连续涨停因子： 在本天前，已连续涨停N次(不算本次),则发出购买信号。
        if flag and self.tradingDaysAble == 1:
            flag = self.fit_tradingDays(today)
        if flag :
            return self.buy_tomorrow()
        else:
            return None