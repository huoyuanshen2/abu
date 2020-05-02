# -*- encoding:utf-8 -*-
"""
    交易可视化模块
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import copy
import datetime

import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from gm.api import history, set_token, get_instrumentinfos
from gm.enum import ADJUST_NONE, ADJUST_PREV
from pandas import DataFrame

from abupy.CoreBu.ABuStore import get_and_store_stock_detail_data, get_and_store_SHSE000001_detail_data
from abupy.UtilBu import ABuPltUtil
from abupy.UtilBu.ABuPltUtil import generatePngName
from ..MarketBu import IndexSymbol, ABuSymbolPd
# from abupy import EMarketDataSplitMode
from abupy.UtilBu.AbuEMAUtil import get_EMA
from ..TradeBu import AbuBenchmark
from abupy.MarketBu.ABuDataCache import load_kline_df
from ..CoreBu import ABuEnv, pd_rolling_mean
from ..CoreBu.ABuEnv import EMarketDataSplitMode
from ..UtilBu import ABuDateUtil
from ..UtilBu.ABuProgress import AbuProgress

# noinspection PyUnresolvedReferences
from ..CoreBu.ABuFixes import range
from ..TradeBu.ABuCapital import AbuCapital

g_enable_his_corr = True
g_enable_his_trade = True

__author__ = '阿布'
__weixin__ = 'abu_quant'


def plot_his_trade(orders, kl_pd):
    """
    可视化绘制AbuOrder对象，绘制交易买入时间，卖出时间，价格，生效因子等
    :param orders: AbuOrder对象序列
    :param kl_pd: 金融时间序列，pd.DataFrame对象
    :return:
    """

    if not g_enable_his_trade:
        return

    # 拿出时间序列中最后一个，做为当前价格
    now_price = kl_pd.iloc[-1].close
    all_pd = kl_pd

    # ipython环境绘制在多个子画布上，普通python环境绘制一个show一个
    draw_multi_ax = ABuEnv.g_is_ipython

    # 根据绘制环境设置子画布数量
    ax_cnt = 1 if not draw_multi_ax else len(orders)
    # 根据子画布数量设置画布大小
    plt.figure(figsize=(14, 8 * ax_cnt))
    fig_dims = (ax_cnt, 1)

    with AbuProgress(len(orders), 0) as pg:
        for index, order in enumerate(orders):

            if (ABuEnv.draw_order_num <= 0):
                break
            ABuEnv.draw_order_num -= 1

            pg.show(index + 1)
            # 迭代所有orders，对每一个AbuOrder对象绘制交易细节
            mask_date = all_pd['date'] == order.buy_date
            st_key = all_pd[mask_date]['key']

            if order.sell_type == 'keep':
                rv_pd = all_pd.iloc[st_key.values[0]:, :]
            else:
                mask_sell_date = all_pd['date'] == order.sell_date
                st_sell_key = all_pd[mask_sell_date]['key']
                rv_pd = all_pd.iloc[st_key.values[0]:st_sell_key.values[0], :]

            if draw_multi_ax:
                # ipython环境绘制在多个子画布上
                plt.subplot2grid(fig_dims, (index, 0))
            # 绘制价格曲线
            plt.plot(all_pd.index, all_pd['close'], label='close')

            try:
                # 填充透明blue, 针对用户一些版本兼容问题进行处理
                plt.fill_between(all_pd.index, 0, all_pd['close'], color='blue', alpha=.18)
                if order.sell_type == 'keep':
                    # 如果单子还没卖出，是否win使用now_price代替sell_price，需＊单子期望的盈利方向
                    order_win = (now_price - order.buy_price) * order.expect_direction > 0
                elif order.sell_type == 'win':
                    order_win = True
                else:
                    order_win = False
                if order_win:
                    # 盈利的使用红色
                    plt.fill_between(rv_pd.index, 0, rv_pd['close'], color='red', alpha=.38)
                else:
                    # 亏损的使用绿色
                    plt.fill_between(rv_pd.index, 0, rv_pd['close'], color='green', alpha=.38)
            except:
                logging.debug('fill_between numpy type not safe!')
            # 格式化买入信息标签
            buy_date_fmt = ABuDateUtil.str_to_datetime(str(order.buy_date), '%Y%m%d')
            buy_tip = 'buy_price:{:.2f}'.format(order.buy_price)

            # 写买入tip信息
            plt.annotate(buy_tip, xy=(buy_date_fmt, all_pd['close'].asof(buy_date_fmt) * 2 / 5),
                         xytext=(buy_date_fmt, all_pd['close'].asof(buy_date_fmt)),
                         arrowprops=dict(facecolor='red'),
                         horizontalalignment='left', verticalalignment='top')

            if order.sell_price is not None:
                # 如果单子卖出，卖出入信息标签使用，收益使用sell_price计算，需＊单子期望的盈利方向
                sell_date_fmt = ABuDateUtil.str_to_datetime(str(order.sell_date), '%Y%m%d')
                pft = (order.sell_price - order.buy_price) * order.buy_cnt * order.expect_direction
                sell_tip = 'sell price:{:.2f}, profit:{:.2f}'.format(order.sell_price, pft)
            else:
                # 如果单子未卖出，卖出入信息标签使用，收益使用now_price计算，需＊单子期望的盈利方向
                sell_date_fmt = ABuDateUtil.str_to_datetime(str(all_pd[-1:]['date'][0]), '%Y%m%d')
                pft = (now_price - order.buy_price) * order.buy_cnt * order.expect_direction
                sell_tip = 'now price:{:.2f}, profit:{:.2f}'.format(now_price, pft)

            # 写卖出tip信息
            plt.annotate(sell_tip, xy=(sell_date_fmt, all_pd['close'].asof(sell_date_fmt) * 2 / 5),
                         xytext=(sell_date_fmt, all_pd['close'].asof(sell_date_fmt)),
                         arrowprops=dict(facecolor='green'),
                         horizontalalignment='left', verticalalignment='top')
            # 写卖出因子信息
            plt.annotate(order.sell_type_extra, xy=(buy_date_fmt, all_pd['close'].asof(sell_date_fmt) / 4),
                         xytext=(buy_date_fmt, all_pd['close'].asof(sell_date_fmt) / 4),
                         arrowprops=dict(facecolor='yellow'),
                         horizontalalignment='left', verticalalignment='top')

            # 写买入因子信息
            if order.buy_factor is not None:
                plt.annotate(order.buy_factor, xy=(buy_date_fmt, all_pd['close'].asof(sell_date_fmt) / 3),
                             xytext=(buy_date_fmt, all_pd['close'].asof(sell_date_fmt) / 3),
                             arrowprops=dict(facecolor='yellow'),
                             horizontalalignment='left', verticalalignment='top')
            # title使用时间序列symbol
            plt.title(order.buy_symbol)
            if not draw_multi_ax:
                # ipython环境绘制在多个子画布上，普通python环境绘制一个show一个
                plt.show()

    plt.show()


def plot_order_myself(order):
    """
    绘制单个订单信息
    :param order: AbuOrder对象序列
    """
    # ax_cnt = 4
    # # 根据子画布数量设置画布大小
    # plt.figure(figsize=(14, 8 * ax_cnt))
    # plot_order_bench_info_myself(order, 0, ax_cnt,start=200,end=100)
    # plot_order_bench_info_myself(order, 1, ax_cnt)
    # plot_order_limit_info_myself(order, 2, ax_cnt,start=100,end=100)
    # plot_order_limit_info_myself(order, 3, ax_cnt)
    # plt.show()
    plot_order_candle_myself(order,start=30,end=15)
    plot_order_jubaopen_myself(order,start=30,end=15)
    # plot_order_wave_myself(order,start=150,end=50)


def plot_order_bench_info_myself(order,index,ax_cnt=4,start=60,end=200):
    """
    绘制单个订单卖出时的大盘信息
    :param order: AbuOrder对象序列
    """
    benchmark = IndexSymbol.SH
    kl_pd = ABuSymbolPd.make_kl_df(benchmark, data_mode=EMarketDataSplitMode.E_DATA_SPLIT_UNDO,
                                        n_folds=2,
                                        start=start, end=end)
    # benchmark = AbuBenchmark(n_folds=2, start=None, end=None)
    # kl_pd = benchmark.kl_pd

    all_pd = kl_pd
    fig_dims = (ax_cnt, 1)
    all_pd['close5'] = pd_rolling_mean(all_pd.close, window=5)
    all_pd['close10'] = pd_rolling_mean(all_pd.close, window=10)
    all_pd['close20'] = pd_rolling_mean(all_pd.close, window=20)
    all_pd['close60'] = pd_rolling_mean(all_pd.close, window=60)
    # 迭代所有orders，对每一个AbuOrder对象绘制交易细节
    mask_date = all_pd['date'] == order.buy_date #买入日期
    st_key = all_pd[mask_date]['key'] #买入日期key
    all_pd = kl_pd[kl_pd.key > st_key.values[0] - start]
    all_pd = all_pd[all_pd.key < st_key.values[0] + end]

    plt.subplot2grid(fig_dims, (index, 0)) #子图位置

    # 绘制价格曲线
    plt.plot(all_pd.index, all_pd['close'], label='close',color='blue')
    plot_order_share_myself(all_pd,st_key, order,'benchMark')
    plt.plot(all_pd.index, all_pd['close5'], label='close5', color='black')
    plt.plot(all_pd.index, all_pd['close10'], label='close10', color='yellow')
    plt.plot(all_pd.index, all_pd['close20'], label='close20', color='red')
    plt.plot(all_pd.index, all_pd['close60'], label='close60', color='green')
    plt.legend(loc='upper left')
    # plt.show()



def plot_order_jubaopen_myself(order,start=60,end=10):
    """
    绘制订单分时图
    :param order: AbuOrder对象序列
    """
    stock_code = order.symbol
    set_token("8e1026d2dfd455be2e1f239e50004b35a481061e")
    data = get_instrumentinfos(symbols=None, exchanges=None, sec_types=1, names=None, fields=None, df=True)
    symbol = data[data.sec_id ==stock_code].symbol.values[0]
    start_date_order = datetime.datetime.strptime(str(order.buy_time)[0:18], "%Y-%m-%d %H:%M:%S").date()
    start_date = start_date_order + datetime.timedelta(days=-1)
    end_date = start_date_order + datetime.timedelta(days=9)



    kl_pd = history(symbol, "60s", start_date, end_date, fields=None, skip_suspended=True,
                    fill_missing=None, adjust=ADJUST_PREV, adjust_end_time='', df=True)
    bench_kl_pd = history('SHSE.000001', "60s", start_date, end_date, fields=None, skip_suspended=True,
                          fill_missing=None, adjust=ADJUST_PREV, adjust_end_time='', df=True)

    bench_kl_pd = bench_kl_pd[bench_kl_pd['bob'].isin(kl_pd['bob'].tolist())]
    bench_kl_pd.index = np.arange(0, len(bench_kl_pd))

    kl_pd['date'] = kl_pd['bob'].apply(lambda x: ABuDateUtil.date_time_str_to_int(str(x)))
    kl_pd['time'] = kl_pd['bob'].apply(lambda x: ABuDateUtil.date_time_str_to_time_str(str(x)))

    kl_pd_time = kl_pd[kl_pd.time == '093000']
    kl_pd_buy_time = kl_pd[kl_pd.bob == order.buy_time]
    kl_pd_sell_time = kl_pd[kl_pd.bob == order.sell_time]

    kl_pd['p_change'] = (kl_pd.close - kl_pd['close'][0]) / kl_pd['close'][0]
    bench_kl_pd['p_change'] = (bench_kl_pd.close - bench_kl_pd['close'][0]) / bench_kl_pd['close'][0]
    kl_pd['p_change_update'] = (kl_pd.p_change - bench_kl_pd.p_change)

    window_volume = 30
    window_close = 30
    kl_pd['p_change_5ma'] = kl_pd.p_change.rolling(window=window_close).mean()
    kl_pd['p_change_update_5ma'] = kl_pd.p_change_update.rolling(window=window_close).mean()
    bench_kl_pd['p_change_5ma'] = bench_kl_pd.p_change.rolling(window=window_close).mean()

    kl_pd['volume_ma'] = kl_pd.volume.rolling(window=window_volume).mean()

    kl_pd['p_change_5ma_up_rate'] = (kl_pd.p_change_5ma - kl_pd.p_change_5ma.shift(5))
    kl_pd['p_change_update_5ma_up_rate'] = (kl_pd.p_change_update_5ma - kl_pd.p_change_update_5ma.shift(5))
    bench_kl_pd['p_change_5ma_up_rate'] = (bench_kl_pd.p_change_5ma - bench_kl_pd.p_change_5ma.shift(5))
    kl_pd['zero_line'] = 0

    kl_pd['volume_ma_up_rate'] = (kl_pd.volume_ma - kl_pd.volume_ma.shift(5))
    kl_pd[ kl_pd['p_change_5ma_up_rate'] > 0.01] = 0.01
    kl_pd[ kl_pd['p_change_5ma_up_rate'] < -0.01] = -0.01
    max_p_change = kl_pd['p_change_5ma_up_rate'].max()
    min_p_change = kl_pd['p_change_5ma_up_rate'].min()
    max_volume = kl_pd['volume_ma_up_rate'].max()
    min_volume = kl_pd['volume_ma_up_rate'].min()

    vs_rate1 = max_p_change / max_volume
    vs_rate2 = min_p_change / min_volume
    vs_rate = vs_rate1 if vs_rate1 >= vs_rate2 else vs_rate2
    kl_pd['volume_ma_up_rate'] = (kl_pd.volume_ma - kl_pd.volume_ma.shift(5)) * vs_rate
    # kl_pd[kl_pd['volume_ma_up_rate'] > 0.0025] = 0.0025
    # kl_pd[kl_pd['volume_ma_up_rate'] < -0.0025] = -0.0025
    # kl_pd['volume_ma_up_rate'] = kl_pd['volume_ma_up_rate']  * 4
    # max_volume = kl_pd['volume_ma_up_rate'].max()
    # min_volume = kl_pd['volume_ma_up_rate'].min()
    #
    # vs_rate1 = max_p_change / max_volume
    # vs_rate2 = min_p_change / min_volume
    # vs_rate = vs_rate1 if vs_rate1 >= vs_rate2 else vs_rate2
    # kl_pd['volume_ma_up_rate'] = (kl_pd.volume_ma - kl_pd.volume_ma.shift(5)) * vs_rate


    title = str(stock_code) + '_' + str(order.buy_time)[0:10]

    # plt.plot(kl_pd.index, kl_pd['p_change'], label='p_change', color='blue') #基础p_change
    # plt.plot(kl_pd.index, bench_kl_pd['p_change'], label='bench_p_change', color='green') #大盘p_change
    # plt.plot(kl_pd.index, kl_pd['p_change_5ma'], label='close60', color='red') #基础p_change均线
    # plt.plot(kl_pd.index, bench_kl_pd['p_change_5ma'], label='close60', color='red') #基础大盘p_change均线
    # plt.plot(kl_pd.index, kl_pd['p_change_update'],'--', label='p_change_update', color='red') #修正后涨跌幅


    plt.plot(kl_pd.index, kl_pd['p_change'], label='p_change', color='blue') #基础p_change
    plt.plot(bench_kl_pd.index, bench_kl_pd['p_change'], label='bench_p_change', color='green') #大盘p_change
    # plt.plot(kl_pd.index, kl_pd['p_change_5ma'], label='close60', color='red') #基础p_change均线
    # plt.plot(kl_pd.index, bench_kl_pd['p_change_5ma'], label='close60', color='red') #基础大盘p_change均线
    # plt.plot(kl_pd.index, kl_pd['p_change_update'],'--', label='p_change_update', color='red') #修正后涨跌幅

    plt.plot(kl_pd.index, kl_pd['zero_line'], label='0_line', color='black')  # 0线
    plt.vlines(kl_pd_time.index, -0.005, 0.005,color="black") #日期分割线
    plt.vlines(kl_pd_buy_time.index, -0.01, 0.01,color="red") #买入时间线
    plt.vlines(kl_pd_sell_time.index, -0.02, 0.02,color="blue") #卖出时间线
    plt.title(title)
    plt.legend(loc='upper left')
    # plt.show()
    png_name = generatePngName(stock_code)
    plt.savefig(png_name)
    plt.close()

    # 获得日分时数据。
    kl_pd = get_and_store_stock_detail_data(stock_code, str(start_date_order))
    kl_pd['zero_line'] = 0
    # plt.plot(kl_pd.index, kl_pd['volume_30ma_up_rate'], label='volume_30ma_up_rate', color='blue') #基础p_change
    plt.plot(kl_pd.index, kl_pd['volume_30ma'], label='volume_30ma', color='blue') #基础p_change
    plt.plot(kl_pd.index, kl_pd['volume_5ma'], label='volume_5ma', color='green') #基础p_change
    plt.plot(kl_pd.index, kl_pd['zero_line'], label='0_line', color='black')  # 0线
    plt.title(title)
    plt.legend(loc='upper left')
    # plt.show()
    png_name = generatePngName(stock_code)
    plt.savefig(png_name)
    plt.close()

    bench_kl_pd = get_and_store_SHSE000001_detail_data(str(start_date_order))


    plt.plot(kl_pd.index, kl_pd['p_change_30ma_up_rate'], label='p_change_30ma_up_rate', color='red')  # 基础均线增长斜率
    # plt.plot(kl_pd.index, kl_pd['p_change_update_5ma_up_rate'], '--', label='close60', color='blue')  # 修正均线增长斜率
    plt.plot(bench_kl_pd.index, bench_kl_pd['p_change_30ma_up_rate'], label='bench_p_change_30ma_up_rate', color='green')  # 大盘增长斜率

    plt.plot(kl_pd.index, kl_pd['zero_line'], label='0_line', color='black')  # 0线
    # plt.plot(kl_pd.index, kl_pd['volume_ma'], label='volume_ma', color='blue') #量均值
    # plt.plot(kl_pd.index, kl_pd['volume_30ma_up_rate'], '--', label='volume_30ma_up_rate', color='blue')  # 量增长斜率
    plt.title(title)
    plt.legend(loc='upper left')
    # plt.show()
    png_name = generatePngName(stock_code)
    plt.savefig(png_name)
    plt.close()

    plt.plot(kl_pd.index, kl_pd['p_change_30ma'], label='p_change_30ma', color='red')  # 基础均线增长斜率
    plt.plot(bench_kl_pd.index, bench_kl_pd['p_change_30ma'], label='bench_p_change_30ma',
             color='green')  # 大盘增长斜率

    plt.plot(kl_pd.index, kl_pd['zero_line'], label='0_line', color='black')  # 0线
    plt.title(title)
    plt.legend(loc='upper left')
    png_name = generatePngName(stock_code)
    plt.savefig(png_name)
    plt.close()
    pass



def plot_order_limit_info_myself(order,index,ax_cnt=4,start=60,end=100):
    """
    绘制单个订单卖出信息最近几天信息
    :param order: AbuOrder对象序列
    """
    stock_code = order.symbol
    from abupy import code_to_symbol
    sh_stock_code = code_to_symbol(stock_code).value
    kl_pd, df_req_start, df_req_end = load_kline_df(sh_stock_code)
    all_pd = kl_pd
    fig_dims = (ax_cnt, 1)
    mask_date = all_pd['date'] == order.buy_date
    st_key = all_pd[mask_date]['key']

    all_pd = all_pd[all_pd.key > st_key.values[0] - start]
    all_pd = all_pd[all_pd.key < st_key.values[0] + end]
    plt.subplot2grid(fig_dims, (index, 0))

    # 绘制价格曲线
    plt.plot(all_pd.index, all_pd['close'], label='close')
    plot_order_share_myself(all_pd, st_key, order, order.symbol)


def plot_order_candle_myself(order,start=60,end=10):
    """
    绘制订单蜡烛图
    :param order: AbuOrder对象序列
    """
    stock_code = order.symbol
    from abupy import code_to_symbol
    sh_stock_code = code_to_symbol(stock_code).value
    all_pd, df_req_start, df_req_end = load_kline_df(sh_stock_code)
    mask_date = all_pd['date'] == order.buy_date
    st_key = all_pd[mask_date]['key']

    if order.sell_type == 'keep':
        rv_pd = all_pd[all_pd.key >= st_key.values[0]]  # 买入到卖出之间的数据
        st_sell_key = all_pd[mask_date]['key']
    else:
        mask_sell_date = all_pd['date'] == order.sell_date  # 卖出日期
        st_sell_key = all_pd[mask_sell_date]['key']  # 卖出日期key
        rv_pd = all_pd[all_pd.key >= st_key.values[0]]
        rv_pd = rv_pd[rv_pd.key < st_sell_key.values[0] + 1]
    tempList = []
    for k in rv_pd.index:
        tempList.append(k)
    if hasattr(order,'bigWave2Close2') :
        all_pd['bigWave2Close2'] = order.bigWave2Close2
        # all_pd['ema10'] = get_EMA(all_pd.close, 10)
        # all_pd['ema60'] = get_EMA(all_pd.close, 60)
        # tempList.append(order.bigWave2Close2Date)

    from abupy import ABuMarketDrawing
    all_pd = all_pd[all_pd.key >= (st_key.values[0] - start)]
    all_pd = all_pd[all_pd.key < (st_sell_key.values[0] + end)]
    all_pd.name = stock_code
    benchmark = IndexSymbol.SH

    bench_pd = ABuSymbolPd.make_kl_df(benchmark, data_mode=EMarketDataSplitMode.E_DATA_SPLIT_UNDO,
                                        n_folds=None,
                                        start=None, end=None)

    bench_mask_date = bench_pd['date'] == order.buy_date  # 买入日期
    bench_st_key = bench_pd[bench_mask_date]['key']  # 买入日期key
    # bench_pd = bench_pd[bench_pd.key > bench_st_key.values[0] - start]
    # bench_pd = bench_pd[bench_pd.key < bench_st_key.values[0] + end]

    if order.sell_type == 'keep':
        rv_pd = bench_pd[bench_pd.key >= bench_st_key.values[0]]  # 买入到卖出之间的数据
        bench_st_sell_key = bench_st_key  # 卖出日期key
    else:
        bench_mask_sell_date = bench_pd['date'] == order.sell_date  # 卖出日期
        bench_st_sell_key = bench_pd[bench_mask_sell_date]['key']  # 卖出日期key
        rv_pd = bench_pd[bench_pd.key >= bench_st_key.values[0]]
        rv_pd = rv_pd[rv_pd.key < bench_st_sell_key.values[0] + 1]

    bench_pd = bench_pd[bench_pd.key >= (bench_st_key.values[0] - start)]
    bench_pd = bench_pd[bench_pd.key <= (bench_st_sell_key.values[0] + end)]

    ABuMarketDrawing.plot_candle_form_klpd(all_pd,html_bk=False,view_indexs= tempList,bench=bench_pd,order=order,save=True)


def plot_order_wave_myself(order,start=100,end=30):
    """
    绘制周期图
    :param order: AbuOrder对象序列
    """
    stockCode = order.symbol
    from abupy import code_to_symbol
    sh_stock_code = code_to_symbol(stockCode).value
    kl_pd, df_req_start, df_req_end = load_kline_df(sh_stock_code)
    mask_date = kl_pd['date'] == order.buy_date
    st_key = kl_pd[mask_date]['key']

    if order.sell_type == 'keep':
        rv_pd = kl_pd[kl_pd.key >= st_key.values[0]]  # 买入到卖出之间的数据
    else:
        mask_sell_date = kl_pd['date'] == order.sell_date  # 卖出日期
        st_sell_key = kl_pd[mask_sell_date]['key']  # 卖出日期key
        rv_pd = kl_pd[kl_pd.key >= st_key.values[0]]
        rv_pd = rv_pd[rv_pd.key < st_sell_key.values[0] + 1]

    from abupy import ABuMarketDrawing
    kl_pd = kl_pd[kl_pd.key >= (st_key.values[0] - start)]
    kl_pd = kl_pd[kl_pd.key < (st_sell_key.values[0] + end)]
    kl_pd.name = stockCode
    benchmark = IndexSymbol.SH
    bench_pd = ABuSymbolPd.make_kl_df(benchmark, data_mode=EMarketDataSplitMode.E_DATA_SPLIT_UNDO,
                                        n_folds=None,
                                        start=None, end=None)

    bench_mask_date = bench_pd['date'] == order.buy_date  # 买入日期
    bench_st_key = bench_pd[bench_mask_date]['key']  # 买入日期key

    if order.sell_type == 'keep':
        rv_pd = bench_pd[bench_pd.key >= bench_st_key.values[0]]  # 买入到卖出之间的数据
    else:
        bench_mask_sell_date = bench_pd['date'] == order.sell_date  # 卖出日期
        bench_st_sell_key = bench_pd[bench_mask_sell_date]['key']  # 卖出日期key
        rv_pd = bench_pd[bench_pd.key >= bench_st_key.values[0]]
        rv_pd = rv_pd[rv_pd.key < bench_st_sell_key.values[0] + 1]

    bench_pd = bench_pd[bench_pd.key >= (bench_st_key.values[0] - start)]
    bench_pd = bench_pd[bench_pd.key < (bench_st_sell_key.values[0] + end)]
    from abupy.UtilBu.AbuJiJinDataUtil import jiJinPlotWave
    jiJinPlotWave(kl_pd=kl_pd, bench_pd=bench_pd, jiJinCodes=[stockCode], windowBuy=18, windowSell=18,
                  poly=50, showWindowBuy=True, showWindowSell=True,keepDays=rv_pd)

def plot_order_share_myself(all_pd, st_key, order, title):
    """
    订单曲线公共绘制方法
    :param order: AbuOrder对象序列
    """
    if order.sell_type == 'keep':
        rv_pd = all_pd[all_pd.key >= st_key.values[0]] # 买入到卖出之间的数据
    else:
        mask_sell_date = all_pd['date'] == order.sell_date  # 卖出日期
        st_sell_key = all_pd[mask_sell_date]['key']  # 卖出日期key
        print(st_sell_key)
        rv_pd = all_pd[all_pd.key >= st_key.values[0]]
        rv_pd = rv_pd[rv_pd.key < st_sell_key.values[0]+1]
    # 拿出时间序列中最后一个，做为当前价格
    now_price = all_pd.iloc[-1].close
    y_min =  all_pd['close'].min()
    try:
        # 填充透明blue, 针对用户一些版本兼容问题进行处理
        plt.fill_between(all_pd.index,y_min, all_pd['close'], color='blue', alpha=.18)
        if order.sell_type == 'keep':
            # 如果单子还没卖出，是否win使用now_price代替sell_price，需＊单子期望的盈利方向
            order_win = (now_price - order.buy_price) * order.expect_direction > 0
        elif order.sell_type == 'win':
            order_win = True
        else:
            order_win = False
        if order_win:
            # 盈利的使用红色
            plt.fill_between(rv_pd.index,y_min, rv_pd['close'], color='red', alpha=.38)
        else:
            # 亏损的使用绿色
            plt.fill_between(rv_pd.index,y_min, rv_pd['close'], color='green', alpha=.38)
    except:
        logging.debug('fill_between numpy type not safe!')
    # 格式化买入信息标签
    buy_date_fmt = ABuDateUtil.str_to_datetime(str(order.buy_date), '%Y%m%d')
    buy_tip = 'buy_price:{:.2f}'.format(order.buy_price)

    position = all_pd['close'].asof(buy_date_fmt) - all_pd['close'].min()
    # 写买入tip信息
    plt.annotate(buy_tip, xy=(buy_date_fmt, all_pd['close'].asof(buy_date_fmt) - position * 1/6),
                 xytext=(buy_date_fmt, all_pd['close'].asof(buy_date_fmt) - position * 1/6),
                 arrowprops=dict(facecolor='red'),
                 horizontalalignment='left', verticalalignment='top')

    if order.sell_price is not None:
        # 如果单子卖出，卖出入信息标签使用，收益使用sell_price计算，需＊单子期望的盈利方向
        sell_date_fmt = ABuDateUtil.str_to_datetime(str(order.sell_date), '%Y%m%d')
        pft = (order.sell_price - order.buy_price) * order.buy_cnt * order.expect_direction
        sell_tip = 's_price:{:.2f},fit:{:.2f},type:{}'.format(order.sell_price, pft,order.sell_type)
    else:
        # 如果单子未卖出，卖出入信息标签使用，收益使用now_price计算，需＊单子期望的盈利方向
        sell_date_fmt = ABuDateUtil.str_to_datetime(str(all_pd[-1:]['date'][0]), '%Y%m%d')
        pft = (now_price - order.buy_price) * order.buy_cnt * order.expect_direction
        sell_tip = 'now price:{:.2f}, profit:{:.2f}, psell_type:{}'.format(now_price, pft,order.sell_type)

    # 写卖出tip信息
    plt.annotate(sell_tip, xy=(sell_date_fmt,all_pd['close'].asof(buy_date_fmt) - position * 3/6),
                 xytext=(sell_date_fmt,all_pd['close'].asof(buy_date_fmt) - position * 3/6),
                 arrowprops=dict(facecolor='green'),
                 horizontalalignment='left', verticalalignment='center')
    # 写卖出因子信息
    plt.annotate(order.sell_type_extra, xy=(buy_date_fmt,all_pd['close'].asof(buy_date_fmt) - position * 4/6),
                 xytext=(buy_date_fmt, all_pd['close'].asof(buy_date_fmt) - position * 4/6),
                 arrowprops=dict(facecolor='yellow'),
                 horizontalalignment='left', verticalalignment='top')

    # 写买入因子信息
    if order.buy_factor is not None:
        plt.annotate(order.buy_factor, xy=(buy_date_fmt,all_pd['close'].asof(buy_date_fmt) - position * 2/6),
                     xytext=(buy_date_fmt, all_pd['close'].asof(buy_date_fmt) - position * 2/6),
                     arrowprops=dict(facecolor='yellow'),
                     horizontalalignment='left', verticalalignment='top')
    plt.title(title)

def plot_capital_info(capital_pd, init_cash=-1):
    """
    资金信息可视化
    :param capital_pd: AbuCapital对象或者AbuCapital对象的capital_pd
    :param init_cash: 初始化cash，如果capital_pd为AbuCapital对象，即从capital_pd获取
    """

    if isinstance(capital_pd, AbuCapital):
        # 如果是AbuCapital对象进行转换
        init_cash = capital_pd.read_cash
        capital_pd = capital_pd.capital_pd

    plt.figure(figsize=(14, 8))
    if init_cash != -1:
        cb_earn = capital_pd['capital_blance'] - init_cash
        try:
            # 从有资金变化开始的loc开始绘制
            # noinspection PyUnresolvedReferences
            cb_earn = cb_earn.loc[cb_earn[cb_earn != 0].index[0]:]
            cb_earn.plot()
            plt.title('capital_blance earn from none zero point')
            plt.show()
            sns.regplot(x=np.arange(0, cb_earn.shape[0]), y=cb_earn.values, marker='+')
            plt.show()
        except Exception as e:
            logging.exception(e)
            capital_pd['capital_blance'].plot()
            plt.title('capital blance')
            plt.show()

    # 为了画出平滑的曲线，取有值的
    cap_cp = copy.deepcopy(capital_pd)
    cap_cp['stocks_blance'][cap_cp['stocks_blance'] <= 0] = np.nan
    cap_cp['stocks_blance'].fillna(method='pad', inplace=True)
    cap_cp['stocks_blance'].dropna(inplace=True)
    cap_cp['stocks_blance'].plot()
    plt.title('stocks blance')
    plt.show()

    try:
        sns.distplot(capital_pd['capital_blance'], kde_kws={"lw": 3, "label": "capital blance kde"})
        plt.show()
    except Exception as e:
        logging.debug(e)
        capital_pd['capital_blance'].plot(kind='kde')
        plt.title('capital blance kde')
        plt.show()


def plot_bk_xd(bk_summary, kl_pd_xd_mean, title=None):
    """根据有bk_summary属性的bk交易因子进行可视化，暂时未迁移完成"""
    plt.figure()
    plt.plot(list(range(0, len(kl_pd_xd_mean))), kl_pd_xd_mean['close'])
    for bk in bk_summary.bk_xd_obj_list:
        plt.hold(True)
        pc = 'r' if bk.break_sucess is True else 'g'
        plt.plot(bk.break_index, kl_pd_xd_mean['close'][bk.break_index], 'ro', markersize=12, markeredgewidth=1.5,
                 markerfacecolor='None', markeredgecolor=pc)
    if title is not None:
        plt.title(title)
    plt.grid(True)


def plot_kp_xd(kp_summary, kl_pd_xd_mean, title=None):
    """根据有bk_summary属性的kp交易因子进行可视化，暂时未迁移完成"""
    plt.figure()
    plt.plot(list(range(0, len(kl_pd_xd_mean))), kl_pd_xd_mean['close'])

    for kp in kp_summary.kp_xd_obj_list:
        plt.hold(True)
        plt.plot(kp.break_index, kl_pd_xd_mean['close'][kp.break_index], 'ro', markersize=8, markeredgewidth=1.5,
                 markerfacecolor='None', markeredgecolor='r')

    if title is not None:
        plt.title(title)
    plt.grid(True)


def plot_result_price(best_result_tuple,count = None):
    """
    可视化绘制回测结果价格
    :param orders: AbuOrder对象序列
    :param kl_pd: 金融时间序列，pd.DataFrame对象
    :return:
    """
    orders_pd = best_result_tuple.orders_pd[0:count if count is not None else 1000]
    for key, order in orders_pd.iterrows():
        buy_date = order.buy_date
        stock_code = order.symbol
        win_status = 'red' if order.sell_type == 'win' else 'green'
        from abupy import code_to_symbol
        stock_code = code_to_symbol(stock_code).value
        kl_pd, df_req_start, df_req_end = load_kline_df(stock_code)
        kl_pd_key = kl_pd[kl_pd.date == buy_date]['key'].values[0]
        kl_pd_back = kl_pd.iloc[kl_pd_key - 30:kl_pd_key + 10, :]  # 获得数据
        kl_pd_back['key2'] = list(range(0, len(kl_pd_back)))

        # buy_price = kl_pd[kl_pd.date == buy_date]['close'].values[0]
        pre_close = kl_pd[kl_pd.key == (kl_pd_key - 1)]['close'].values[0]

        rate = 100 / pre_close
        kl_pd_back['close2'] = kl_pd_back['close'].map(lambda x: x * rate)

        # plt.plot(kl_pd_back.key2, kl_pd_back.p_change, label='p_change',color = win_status)
        plt.plot(kl_pd_back.key2, kl_pd_back.close2, label='p_change', color=win_status)
        # plt.show()
    plt.show()


def plot_result_volumn(best_result_tuple, count=None):
    """
    可视化绘制回测结果交易量
    :param orders: AbuOrder对象序列
    :param kl_pd: 金融时间序列，pd.DataFrame对象
    :return:
    """
    orders_pd = best_result_tuple.orders_pd[0:count if count is not None else 1000]
    for key, order in orders_pd.iterrows():
        buy_date = order.buy_date
        stock_code = order.symbol
        win_status = 'red' if order.sell_type == 'win' else 'green'
        from abupy import code_to_symbol
        stock_code = code_to_symbol(stock_code).value
        kl_pd, df_req_start, df_req_end = load_kline_df(stock_code)
        kl_pd_key = kl_pd[kl_pd.date == buy_date]['key'].values[0]
        kl_pd_back = kl_pd.iloc[kl_pd_key - 30:kl_pd_key + 10, :]  # 获得数据
        kl_pd_back['key2'] = list(range(0, len(kl_pd_back)))
        pre_data = kl_pd[kl_pd.key == (kl_pd_key - 1)]['volume'].values[0]
        rate = 100 / pre_data
        kl_pd_back['volume2'] = kl_pd_back['volume'].map(lambda x: 500 if ((x * rate) > 500) else (x * rate))
        plt.plot(kl_pd_back.key2, kl_pd_back.volume2, label='p_change', color=win_status)
    plt.show()



def plot_result_volumn_sum(best_result_tuple,count=None):
    """
    可视化绘制回测结果累计交易量
    :param count: 要绘制的数量
    """
    orders_pd = best_result_tuple.orders_pd[0:count if count is not None else 1000]
    for key, order in orders_pd.iterrows():
        buy_date = order.buy_date
        stock_code = order.symbol
        win_status = 'red' if order.sell_type == 'win' else 'green'
        from abupy import code_to_symbol
        stock_code = code_to_symbol(stock_code).value
        kl_pd, df_req_start, df_req_end = load_kline_df(stock_code)
        kl_pd_key = kl_pd[kl_pd.date == buy_date]['key'].values[0]
        kl_pd_back = kl_pd.iloc[kl_pd_key - 10:kl_pd_key + 1, :]  # 获得数据
        kl_pd_back['key2'] = list(range(0, len(kl_pd_back)))
        kl_pd_back['volume_sum'] = kl_pd_back['volume'].cumsum()

        pre_data = kl_pd_back[kl_pd.key == (kl_pd_key - 1)]['volume_sum'].values[0]
        rate = 100 / pre_data
        kl_pd_back['volume2'] = kl_pd_back['volume_sum'].map(lambda x: 500 if ((x * rate) > 500) else (x * rate))

        plt.plot(kl_pd_back.key2, kl_pd_back.volume2, label='p_change', color=win_status)
        # plt.show()
    plt.show()

def plot_result_pearsonr(best_result_tuple,count = None):
    """
    绘制目标股票行业相关性最高的N只股票最近M天的走势,
    count: order数量
    :return:
    """
    targetCountRate = 5 #想关度股票数量

    orders_pd = best_result_tuple.orders_pd[0:count if count is not None else 1000]
    for key, order in orders_pd.iterrows():
        buy_date = order.buy_date
        stock_code = order.symbol[-6:len(order.symbol)]
        win_status = 'red' if order.sell_type == 'win' else 'green'
        plot_line_percent(stock_code, buy_date, start=6, end=2, win_status=win_status)

        stockIndustryNames = ABuEnv.industry_data['c_name'][ABuEnv.industry_data.code == int(stock_code)]
        for _, stockIndustryName in stockIndustryNames.items():
            industryPearsonrData = ABuEnv.industryPearsonrDataDic[stockIndustryName]
            industryPearsonr4One = industryPearsonrData[stock_code]
            industryPearsonr4OneOrderLimit = industryPearsonr4One.sort_values(ascending=False)[
                                             1:targetCountRate + 1]  # 取相关度最大的targetCountRate只
            industryData = ABuEnv.industryPchangeAllDataDic[stockIndustryName]
            industryTodayData = industryData[industryData.date == buy_date][0:1]
            for stockCodeTemp in industryPearsonr4OneOrderLimit.index:
                win_status2 = 'yellow' if order.sell_type == 'win' else 'blue'
                plot_line_percent(stockCodeTemp, buy_date, start=6, end=2, win_status=win_status2)
        # plt.show()
    plt.show()


def plot_line_percent(stock_code, buy_date,start=6,end=2,win_status='green'):
    """
    百分比方式绘图
    :return:
    """
    from abupy import code_to_symbol
    stock_code_str = code_to_symbol(stock_code).value
    kl_pd, df_req_start, df_req_end = load_kline_df(stock_code_str)
    temp = kl_pd[kl_pd.date == buy_date]['key']

    if kl_pd[kl_pd.date == buy_date]['key'].size == 0:
        return
    kl_pd_key = kl_pd[kl_pd.date == buy_date]['key'].values[0]

    kl_pd_back = kl_pd.iloc[kl_pd_key - start:kl_pd_key + end, :]  # 获得数据
    kl_pd_back['key2'] = list(range(0, len(kl_pd_back)))

    pre_close = kl_pd[kl_pd.key == (kl_pd_key - 1)]['close'].values[0]  # 前一天收盘价
    rate = 100 / pre_close
    kl_pd_back['close2'] = kl_pd_back['close'].map(lambda x: x * rate)
    plt.plot(kl_pd_back.key2, kl_pd_back.close2, label='p_change', color=win_status)
