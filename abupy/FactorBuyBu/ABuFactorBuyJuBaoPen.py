# -*- encoding:utf-8 -*-
"""
    买入择时示例因子：聚宝盆形态
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import time, datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abupy.CoreBu import ABuEnv, pd_rolling_mean
from abupy.CoreBu.ABuStore import get_and_store_stock_detail_data, get_and_store_SHSE000001_detail_data
from abupy.UtilBu import ABuRegUtil, ABuDateUtil
from .ABuFactorBuyBase import AbuFactorBuyXD, BuyCallMixin
from ..IndicatorBu.ABuNDMa import calc_ma_from_prices
from ..CoreBu.ABuPdHelper import pd_resample
from ..TLineBu.ABuTL import AbuTLine
import abupy.UtilBu.ABuRegUtil
__author__ = '阿布'
__weixin__ = 'abu_quant'


# noinspection PyAttributeOutsideInit
class AbuJuBaoPen(AbuFactorBuyXD, BuyCallMixin):
    """聚宝盆"""
    def _init_self(self, **kwargs):
        #是否启用主要因子，默认关闭。
        self.juBaoPenAble = kwargs.pop('juBaoPenAble', 0)
        self.juBaoPenStatus = 0  # 默认订单状态
        self.downDays = kwargs.pop('downDays', 3)   #连跌次数
        self.downRate = kwargs.pop('downRate', -0.1)  # 连续跌幅
        self.openJClose = kwargs.pop('openJClose', 0.05) #实体落差距离比例
        self.maxValuePositionStart = kwargs.pop('maxValuePositionStart', 0)
        self.maxValuePositionEnd = kwargs.pop('maxValuePositionEnd', 50)
        self.buyPrice = kwargs.pop('buyPrice', 10000)
        self.fit_xd = kwargs.pop('fit_xd', 1) #是否进行xd历史数据处理。用于过滤k线形态。
        self.fit_bench = kwargs.pop('fit_bench', 0) #是否进行大盘历史数据处理。用于过滤k线形态。

        # 峰落差，倒V的(最高-最低)/最低 的值
        self.peak_atr = kwargs.pop('peak_atr', 0.1)

        self.skip_days_value = kwargs.pop('skip_days_value', 0) #买入因子冷却间隔



        super(AbuJuBaoPen, self)._init_self(**kwargs)
        # 在输出生成的orders_pd中显示的名字
        self.factor_name = '{}:Days={},GTX={},Rate={}'.format(self.__class__.__name__, self.benchmark,
                                                                                    self.downDays, self.downDays)
    def fit_month(self, today):
        pass


    def fit_bench2(self,today):
        '''
        30日大盘均线向上，通过。
        :param today:
        :return:
        '''
        benchmark_df = self.benchmark.kl_pd
        # 拿出大盘的今天
        benchmark_today = benchmark_df[benchmark_df.date == today.date]
        if benchmark_today.empty:
            return False
        end_key = int(benchmark_today.iloc[0].key)
        start_key = end_key - 60
        if start_key < 0:
            return False
        # 使用切片切出从今天开始向前N天的数据
        benchmark_xd = benchmark_df[start_key:end_key + 1]
        window_close = 30
        benchmark_xd['p_change_30ma'] = benchmark_xd.p_change.rolling(window=window_close).mean()
        benchmark_xd['p_change_30ma_up_rate'] = (benchmark_xd.p_change_30ma - benchmark_xd.p_change_30ma.shift(5))

        if  benchmark_xd.iloc[-1]['p_change_30ma_up_rate'] > 0:
            return True
        else :
            return False
        pass
    def fit_xd2(self, today): #处理前期数据

        kl_pd = self.xd_kl
        min_value = kl_pd['close'].min()
        max_value = kl_pd['close'].max()

        # 峰值周期参考20日内数据
        k_pd = self.kl_pd.iloc[self.today_ind-20:self.today_ind]
        k_min_value = k_pd['low'].min()
        k_max_value = k_pd['close'].max()
        # 确定正值周期内的最大值处于历史较低阈值。减少出现跌停的概率。
        if (k_max_value - min_value)/(max_value-min_value) >= 0.3 :
            return False

        #提前判断今天的数据是否可能符合条件。提高执行效率。
        if (today.low - k_min_value) / self.yesterday.close > 0.005:
            return False

        min_key = k_pd[k_pd.low == k_min_value].key.values[0]
        max_key = k_pd[k_pd.close == k_max_value].key.values[0]
        # 确定峰值的落差大小，是探针形成前提。
        if (k_max_value - k_min_value)/k_min_value <= self.peak_atr:
            return False
        # 今天的最低值能触到支撑线
        # 前一天的收盘价距离阈值的距离。太近的话，今天的探针形态不好。
        #今天开盘价距离阈值的距离：形成探针的前提。
        if not ((self.yesterday.close - k_min_value)/self.yesterday.close <= 0.1 and \
                (self.yesterday.close - k_min_value)/self.yesterday.close >= 0.04 and \
                (today.open - k_min_value)/self.yesterday.close >= 0.02):
            return False
        #确定峰值的位置居中
        if min_key >= self.today_ind - 16 and min_key <= self.today_ind -7:
            if max_key >=  self.today_ind - 8  and max_key <= self.today_ind - 3:
                if self.yesterday.close < self.bf_yesterday.close:
                    return True
                else:
                    return False

        return False
        stockCode = today.stockCode
        kl_pd = self.xd_kl
        # min_value = kl_pd['close'].min()
        # max_value = kl_pd['close'].max()

        # window_close = 30
        # kl_pd['p_change_30ma'] = kl_pd.p_change.rolling(window=window_close).mean()
        # kl_pd['p_change_30ma_up_rate'] = (kl_pd.p_change_30ma - kl_pd.p_change_30ma.shift(5))

        # temp = kl_pd.iloc[-1]['p_change_30ma_up_rate']

        # if today.close <= ((max_value - min_value)/5 + min_value ) and kl_pd.iloc[-1]['p_change_30ma_up_rate'] > 0 :
        # if today.close <= ((max_value - min_value)/5 + min_value ) and self.yesterday.close < self.bf_yesterday.close :



    def fit_juBaoPen(self, today):
        self.today_ind = int(today.key)
        kl_pd = self.kl_pd

        stockCode = today.stockCode

        date_str = time.strftime("%Y-%m-%d", time.strptime(str(today.date), "%Y%m%d"))
        kl_pd = get_and_store_stock_detail_data(stockCode, date_str)

        # bench_kl_pd = get_and_store_SHSE000001_detail_data(date_str)
        if kl_pd is None or len(kl_pd) <= 200:
            return False

        k_pd = self.kl_pd.iloc[self.today_ind - 20:self.today_ind]
        k_min_value = k_pd['low'].min()
        close2Date = k_pd[k_pd.low == k_min_value].date.values[0]
        # k_max_value = k_pd['close'].max()

        len_pd = np.arange(10,len(kl_pd),1)
        flag = False
        for i in len_pd:
            check_pd = kl_pd.iloc[i]
            check_pd_1 = kl_pd.iloc[i - 1]
            check_pd_2 = kl_pd.iloc[i - 2]
            check_pd_3 = kl_pd.iloc[i - 3]
            check_pd_4 = kl_pd.iloc[i - 4]
            # 1、当前价格低于阈值。2、价格处于回调阶段.3、下探或回调的速度要求，形成针形。
            if (check_pd_1.low - k_min_value )/k_min_value <= 0.002 \
                    and (check_pd_1.low <= check_pd.low  ) :
                    # and (check_pd.low - check_pd_2.low)/k_min_value  >= 0.005 :
                #          or (check_pd_4.close - check_pd_2.close)/self.yesterday.close >= 0.005):
                self.buyPrice = check_pd.close
                self.buyTime = check_pd.bob
                self.close2 = k_min_value
                self.close2Date = today.date

                # 买入后接下来几天的数据
                day01 = self.kl_pd.iloc[self.today_ind + 1]
                day02 = self.kl_pd.iloc[self.today_ind + 2]
                day03 = self.kl_pd.iloc[self.today_ind + 3]
                day04 = self.kl_pd.iloc[self.today_ind + 4]
                day05 = self.kl_pd.iloc[self.today_ind + 5]
                day06 = self.kl_pd.iloc[self.today_ind + 6]
                p_change_01 = round((day01.close - today.close) / today.close * 100, 2)
                p_change_02 = round((day02.close - today.close) / today.close * 100, 2)
                p_change_03 = round((day03.close - today.close) / today.close * 100, 2)
                p_change_04 = round((day04.close - today.close) / today.close * 100, 2)
                p_change_05 = round((day05.close - today.close) / today.close * 100, 2)
                p_change_06 = round((day06.close - today.close) / today.close * 100, 2)

                self.feature_state = str(i) + '_' + str(i) + '_' + str(i)[0:2] + '_' \
                                     + str(i)[0:10] + '_' + str(today.volume) + '_' + str(today.amount) \
                                     + '_' + str(p_change_01) + '_' + str(p_change_02) + '_' + str(p_change_03) \
                                     + '_' + str(p_change_04) + '_' + str(p_change_05) + '_' + str(p_change_06)
                return True
        return False
        # if (len(bench_kl_pd) - len(kl_pd)) < 0:
        #     print("(len(bench_kl_pd) - len(kl_pd)) < 0 ,exit.")

        # bench_kl_pd = bench_kl_pd[bench_kl_pd['bob'].isin(kl_pd['bob'].tolist())]
        # bench_kl_pd.index = np.arange(0, len(bench_kl_pd))

        kl_pd_time = kl_pd[kl_pd.time == 93000]
        # bench_kl_pd['p_change'] = (bench_kl_pd.close - bench_kl_pd['close'][0]) / bench_kl_pd['close'][0]
        # window_volume = 30
        # window_close = 30
        # bench_kl_pd['p_change_30ma'] = bench_kl_pd.p_change.rolling(window=window_close).mean()
        # bench_kl_pd['p_change_30ma_up_rate'] = (bench_kl_pd.p_change_30ma - bench_kl_pd.p_change_30ma.shift(5))
        kl_pd['zero_line'] = 0

        # get_ready_flag = 0
        # check_start_i = None
        # for i in kl_pd.index:
        #     if i < 60:
        #         continue
        #     start = i - 10
        #     # if bench_kl_pd['p_change_30ma_up_rate'][start:i].max() < 0 \
        #     if kl_pd['p_change_30ma_up_rate'][start:i].min() > 0 :
        #             # and kl_pd['volume_30ma_up_rate'][start:i].min() > 0 :
        #         get_ready_flag = 1
        #         check_start_i = i
        #
        #         # plt.plot(kl_pd.index, kl_pd['p_change_30ma_up_rate'], label='close60', color='red')  # 基础均线增长斜率
        #         # plt.plot(kl_pd.index, kl_pd['p_change_update_30ma_up_rate'], '--', label='close60',
        #         #          color='blue')  # 修正均线增长斜率
        #         # plt.plot(bench_kl_pd.index, bench_kl_pd['p_change_30ma_up_rate'], label='close60',
        #         #          color='green')  # 大盘增长斜率
        #         #
        #         # plt.plot(kl_pd.index, kl_pd['zero_line'], label='0_line', color='black')  # 0线
        #         # # plt.vlines(kl_pd_time.index, middle_line, middle_line+0.2,color="black") #日期分割线
        #         # # plt.plot(kl_pd.index, kl_pd['volume_30ma'], label='volume_30ma', color='blue') #量均值
        #         # plt.plot(kl_pd.index, kl_pd['volume_30ma_up_rate'], '--', label='up_rate', color='red')  # 量增长斜率
        #         # plt.show()
        #         break
        #     else :
        #         continue

        #上涨的长度，针对不同股票变化不同。理论上只要在一定长度内，符合涨幅形态要求，就是有效的。
        #上涨的平滑度，用线性拟合？用直线均值求均差方法。

        if  1 == 1:
            kl_pd_end = kl_pd[kl_pd.time <= 144000]
            end_index = kl_pd_end.iloc[-1:].index[0]
            len_pd = np.arange(80,200,10)
            flag = False
            for i in len_pd:
                check_pd = kl_pd.iloc[end_index-i:end_index]
                len0 = len(check_pd)
                if len0  <= 40:
                    return False

                y = check_pd.p_change
                rolling_window = int(math.ceil(len(y) / 4))
                y_roll_mean = pd_rolling_mean(y, window=rolling_window, min_periods=1)
                from sklearn import metrics
                rdistance_mean = np.sqrt(metrics.mean_squared_error(y, y_roll_mean))

                len1 = len(check_pd[check_pd['p_change_30ma_up_rate'] > 0])
                len2 = len(check_pd[check_pd['volume_30ma_up_rate'] > 0])
                price_count = len0
                price_rate = len1 / len0 * 100
                volume_rate = len2 / len0 * 100

                if price_rate >= 70 \
                        and volume_rate >= 50  and price_count > 60:
                    self.buyPrice = kl_pd['close'][end_index]
                    self.buyTime = kl_pd['bob'][end_index]
                    self.feature_state = str(price_count) + '_' + str(price_rate)[0:2] + '_' + str(volume_rate)[0:2]+'_'\
                                         +str(rdistance_mean)[0:10]+'_'+str(today.volume)+'_'+str(today.amount)\
                                         +'_'+str(p_change_01)+'_'+str(p_change_02)+'_'+str(p_change_03)\
                                         +'_'+str(p_change_04)+'_'+str(p_change_05)+'_'+str(p_change_06)
                    flag = True
                else:
                    break
            if flag:
                return True
        return False


    def fit_day(self, today):
        int_date = today.date
        # 过滤天数，一般是大节假日前
        if 20200103 < int_date < 20200212  or 20190419 < int_date < 20190510 \
                or 20190120 < int_date < 20190218 or 20181220 < int_date < 20190105  \
                or 20191220 < int_date < 20200105:
            return None
        # 只在指定日期运行
        # if int_date != 20200324:
        #     return None
        # 只处理某只股票
        # stockCode = today.stockCode
        # if stockCode != '300298':
        #     return None

        flag = True
        if flag and self.fit_xd == 1:
            flag = self.fit_xd2(today)
        if flag and self.fit_bench == 1:
            pass
            # flag = self.fit_bench2(today)
        if flag and self.juBaoPenAble == 1:
            # pass
            flag = self.fit_juBaoPen(today)
            # flag = self.fit_juBaoPen(today)
        #6、大盘走势因子： 最近N天取角度，大于X（bench_xd）值，则发出购买信号。
        if flag :
            # return self.buy_tomorrow()
            return self.buy_today(usePrice=1)
        else:
            return None