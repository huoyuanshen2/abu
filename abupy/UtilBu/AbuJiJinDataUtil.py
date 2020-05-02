# -*- encoding:utf-8 -*-
"""
    abupy中基金数据获取
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime

from  gm.api import *

import abupy
# from abupy.SimilarBu.ABuCorrcoef import corr_matrix
# from ..SimilarBu.ABuCorrcoef import corr_matrix
from abupy.UtilBu import ABuDateUtil
import pandas as pd
import matplotlib.pyplot as plt
# from abupy.MarketBu import  IndexSymbol, ABuSymbolPd
from abupy.CoreBu.ABuEnv import   EMarketDataSplitMode, EMarketDataFetchMode, EMarketSourceType, \
    EMarketTargetType
from abupy.CoreBu import ABuEnv
import numpy as np
from selenium.webdriver.support.ui import WebDriverWait
from selenium import webdriver
from abupy.UtilBu.ABuRegUtil import regress_xy_polynomial
from abupy.UtilBu.AbuEMAUtil import get_EMA

ABuEnv.g_data_fetch_mode = EMarketDataFetchMode.E_DATA_FETCH_NORMAL  #强制本地，可多线程
# ABuEnv.g_data_fetch_mode = EMarketDataFetchMode.E_DATA_FETCH_FORCE_NET  #强制本地，可多线程

ABuEnv.g_market_source = EMarketSourceType.E_MARKET_SOURCE_tx  #作用同上点击效果。腾讯数据源(美股，A股，港股)
ABuEnv.g_market_target = EMarketTargetType.E_MARKET_TARGET_CN
__author__ = '阿布'
__weixin__ = 'abu_quant'

# 初始化函数
def initSpider(stockCode):
    driver = webdriver.PhantomJS(executable_path=r"D:\phantomjs-2.1.1-windows\bin\phantomjs.exe")
    driver.get("http://fund.eastmoney.com/f10/jjjz_" + stockCode + ".html")  # 要抓取的网页地址
    # 找到"下一页"按钮,就可以得到它前面的一个label,就是总页数
    getPage_text = driver.find_element_by_id("pagebar").find_element_by_xpath(
        "div[@class='pagebtns']/label[text()='下一页']/preceding-sibling::label[1]").get_attribute("innerHTML")
    # 得到总共有多少页
    total_page = int("".join(filter(str.isdigit, getPage_text)))
    # 返回
    return (driver, total_page)

# 获取html内容
def getData(pageNo, driver):
    tonum = driver.find_element_by_id("pagebar").find_element_by_xpath(
        "div[@class='pagebtns']/input[@class='pnum']")  # 得到 页码文本框
    jumpbtn = driver.find_element_by_id("pagebar").find_element_by_xpath(
        "div[@class='pagebtns']/input[@class='pgo']")  # 跳转到按钮
    tonum.clear()  # 第x页 输入框
    tonum.send_keys(str(pageNo))  # 去第x页
    jumpbtn.click()  # 点击按钮
    # 抓取
    WebDriverWait(driver, 20).until(lambda driver: driver.find_element_by_id("pagebar").
                                    find_element_by_xpath(
        "div[@class='pagebtns']/label[@value={0} and @class='cur']".format(pageNo)) != None)

    rows = getTableData(driver.find_element_by_id("jztable"))
    df = pd.DataFrame(rows, columns=['date_old', 'dwjz', 'ljjz', 'p_change', 'buy_state', 'sell_state', 'share_state'])
    return df

def getTableData(driver):
    '''
    获取表格数据
    colNO:存在合并行时，强制指定表格列数
    '''
    tbody = driver.find_element(by="tag name", value='tbody')
    rows = tbody.find_elements(by="tag name", value="tr")  # 行集合
    maxrowCount = len(rows)  # 表格有效数据行数

    cols = tbody.find_elements(by="tag name", value="td");
    maxcolCount = len(cols) / len(rows)

    lists = [[] for i in range(maxrowCount)]
    i = 0
    while i < maxrowCount:
        j = 0;  # 初始化列
        tds = rows[i].find_elements(by="tag name", value='td')
        while j < maxcolCount:
            coe = tds[j].get_attribute("innerHTML")
            lists[i].append(coe)
            j += 1
        i += 1
    return lists

# 开始抓取函数
def getJiJinData(stockCode,pages=None):
    '''
    ['date', 'dwjz单位净值', 'ljjz累计净值', 'dayRate日增长率', 'buy_state申购状态', 'sell_state赎回状态', 'share_state分红送配']
    :param stockCode:
    :return:
    '''
    (driver, total_page) = initSpider(stockCode)
    if pages is not None:
        total_page = pages
    pageList = range(1, int(total_page) + 1)
    kl_pd = pd.DataFrame()
    for i in pageList:
        df = getData(i, driver)
        kl_pd = pd.concat([df, kl_pd])

    kl_pd.sort_values(by='date_old', ascending=True, inplace=True)
    dates_pd = pd.to_datetime(kl_pd.date_old)
    kl_pd.set_index(dates_pd, drop=True, append=False, inplace=True, verify_integrity=False)
    kl_pd.index.name='index_name'
    kl_pd['date'] = kl_pd['date_old'].apply(lambda x: ABuDateUtil.date_str_to_int(str(x)))
    kl_pd['dwjz'] = kl_pd['dwjz'].astype(float)
    kl_pd['ljjz'] = kl_pd['ljjz'].astype(float)

    kl_pd = kl_pd[kl_pd.p_change != '--']
    kl_pd['p_change'] = kl_pd['p_change'].apply(lambda x: str(x).replace('%','') )
    kl_pd['p_change'] = kl_pd['p_change'].astype(float)
    kl_pd['stockCode'] = stockCode
    kl_pd.name = stockCode

    kl_pd['volume'] = 0
    kl_pd['close'] = kl_pd['dwjz']
    kl_pd['open'] = 0
    kl_pd['high'] = 0
    kl_pd['low'] = 0
    kl_pd['pre_close'] = 0
    return kl_pd

# 开始抓取基金数据——掘金量化方法
def getJiJinData4JueJin(stockCode,startDate,endDate):
    '''
    '''
    if stockCode == '000001':
        symbols='SHSE.'+str(stockCode)
        sec_types = None
    else:
        symbols = ['SHSE.' + str(stockCode),'SZSE.' + str(stockCode)]
        sec_types=None
    data = get_instrumentinfos(symbols=symbols, exchanges=None, sec_types=sec_types, names=None, fields=None, df=True)
    # data = data[data.sec_name.str.contains('LOF')]
    if len(data)== 0:
        return None
    symbolPD = data[data.sec_id == stockCode]
    if len(symbolPD)== 0:
        return None
    symbol = symbolPD.symbol.values[0]
    kl_pd = history(symbol, "1d", startDate, endDate, fields=None, skip_suspended=False,
                    fill_missing=None, adjust=ADJUST_PREV , adjust_end_time='', df=True)
    if len(kl_pd)== 0:
        return kl_pd
    if 'bob' not in kl_pd.columns :
        print("bob not in kl_pd")
        print("stockCode:{}".format(stockCode))
    kl_pd['date'] = kl_pd['bob'].apply(lambda x: ABuDateUtil.date_str_to_int(str(x)))
    kl_pd['bob2'] = pd.to_datetime(kl_pd.bob)
    kl_pd.set_index(kl_pd['bob2'].apply(lambda x: x.replace(tzinfo=None)), drop=True, append=False, inplace=True, verify_integrity=False)
    kl_pd.index.name='index_name'
    kl_pd['stockCode'] = stockCode
    kl_pd['p_change'] = (kl_pd.close - kl_pd.pre_close)/kl_pd.pre_close*100
    a = pd.DataFrame(kl_pd, columns=['stockCode','p_change','open','close','high','low'])
    # kl_pd['date'] = kl_pd['date_old'].apply(lambda x: ABuDateUtil.date_str_to_int(str(x)))
    # kl_pd['dwjz'] = kl_pd['dwjz'].astype(float)
    # kl_pd['ljjz'] = kl_pd['ljjz'].astype(float)
    # kl_pd = kl_pd[kl_pd.dayRate != '--']
    # kl_pd['dayRate'] = kl_pd['dayRate'].apply(lambda x: str(x).replace('%','') )
    kl_pd.name = stockCode
    return kl_pd


def jiJinPlot(jiJinCodes=None, startDate=None, endDate=None, windowBuy=7, windowSell=7,
              poly=60, isJiJin=True, showJiJinOldPchange=True,showWindowBuy=True, showWindowSell=False, concept=None, kl_pd_manager=None,showMulti=False):
    '''
    基金排除大盘背景后数据绘图与买入点计算
    '''
    windowSell = windowBuy
    bench_pd = ABuEnv.benchmark
    if isJiJin :
        pd_dic = kl_pd_manager.pick_kl_pd_dict['pick_time']
        for key in pd_dic.keys():
            if pd_dic[key] is None:
                continue
            ax_cnt = 3
            plt.figure(figsize=(14, 8 * ax_cnt))
            jiJinPlotWave(kl_pd=pd_dic[key], bench_pd=bench_pd, jiJinCodes=jiJinCodes, windowBuy=windowBuy,
                          windowSell=windowSell,
                          poly=50,showJiJinOldPchangeWave=True, showWindowBuy=False, showWindowSell=False,
                          pltShow=False,index=0, ax_cnt=ax_cnt)
            jiJinPlotWave(kl_pd=pd_dic[key], bench_pd=bench_pd, jiJinCodes=jiJinCodes, windowBuy=windowBuy,
                          windowSell=windowSell,
                          poly=50, showJiJinOldPchangeWave=False, showWindowBuy=True, showWindowSell=False,
                          pltShow=False, index=1, ax_cnt=ax_cnt)
            jiJinPlotWave(kl_pd=pd_dic[key], bench_pd=bench_pd, jiJinCodes=jiJinCodes, windowBuy=windowBuy,
                          windowSell=windowSell,
                          poly=50, showJiJinOldPchangeWave=False, showWindowBuy=False, showWindowSell=False,showBenchWave=True,
                          pltShow=False, index=2, ax_cnt=ax_cnt)
            plt.show()
        plt.show()
        return

    else : #处理股票数据
        kl_pd = ABuEnv.industryPchangeAllDataDic[concept].copy()
        temp_pd = kl_pd.drop(columns=['date'])
        kl_pd['p_change'] = temp_pd.mean(1)
        jiJinPlotWave(kl_pd=kl_pd, bench_pd=bench_pd, jiJinCodes=jiJinCodes, windowBuy=windowBuy, windowSell=windowSell,
                     poly=50, showWindowBuy=showWindowBuy, showWindowSell=showWindowSell)

def jiJinPlotWave(kl_pd=None,bench_pd=None,jiJinCodes=None, windowBuy=7, windowSell=7,
                  poly=50, showJiJinClose=False,showBenchClose=False, showJiJinOldPchangeWave=False,showWindowBuy=True, showWindowSell=False,
                  showBenchWave=False,keepDays=None ,pltShow=False,index=None,ax_cnt=None):
    if index is not None:
        fig_dims = (ax_cnt, 1)
        plt.subplot2grid(fig_dims, (index, 0))  # 子图位置

    kl_pd_len = len(kl_pd)
    bench_pd_len = len(bench_pd)
    if kl_pd_len != bench_pd_len:
        kl_pd['dayRateUpdated'] = kl_pd.p_change
    else:
        kl_pd['dayRateUpdated'] = kl_pd.p_change - bench_pd.p_change

    kl_pd['dayRateSum_old'] = kl_pd.dayRateUpdated.rolling(window=windowBuy).sum()
    kl_pd['dayRateSum_old'].fillna(value=0, inplace=True)
    kl_pd.sort_values(by='date', ascending=False, inplace=True)
    kl_pd['dayRateSum'] = kl_pd.dayRateUpdated.rolling(window=windowSell).sum()
    kl_pd['dayRateSum'].fillna(value=0, inplace=True)
    kl_pd.sort_values(by='date', ascending=True, inplace=True)
    legendList = ['date']
    if showJiJinClose:
        plt.plot(kl_pd.index, kl_pd.close, color='blue')
        kl_pd['ema20'] = get_EMA(kl_pd.close, 20)
        kl_pd['ema60'] = get_EMA(kl_pd.close, 60)
        plt.plot(kl_pd.index, kl_pd.ema20, color='green')
        plt.plot(kl_pd.index, kl_pd.ema60, color='red')
        plt.title(jiJinCodes[0])
        legendList.append('close')
        legendList.append('ema20')
        legendList.append('ema60')
    if showBenchClose:
        plt.plot(bench_pd.index, bench_pd.close, color='blue')
        bench_pd['ema20'] = get_EMA(bench_pd.close, 20)
        bench_pd['ema60'] = get_EMA(bench_pd.close, 60)
        plt.plot(bench_pd.index, bench_pd.ema20, color='green')
        plt.plot(bench_pd.index, bench_pd.ema60, color='red')
        plt.title('000001')
        legendList.append('close')
        legendList.append('ema20')
        legendList.append('ema60')

    if showBenchWave:
        bench_pd['dayRateSum_old'] = bench_pd.p_change.rolling(window=windowBuy).sum()
        bench_pd['dayRateSum_old'].fillna(value=0, inplace=True)
        plt.plot(bench_pd.index, bench_pd['dayRateSum_old'], 'o', markersize=3, markeredgewidth=1,
                 markerfacecolor='None', label='p_change2', color='red')
        x = np.arange(0, len( bench_pd.index))
        y_fit = regress_xy_polynomial(x, bench_pd.dayRateSum_old, poly=poly, zoom=False, show=False)
        plt.plot(bench_pd.index, y_fit, color='blue')
        plt.title('000001')
        legendList.append('BenchWave')
        legendList.append('BenchWave')

    if not showJiJinClose and not showBenchClose:
        plt.plot(kl_pd.index, kl_pd.p_change - kl_pd.p_change, '.', markersize=3, markeredgewidth=1,
             markerfacecolor='None', label='p_change2', color='black')  # 日期


    x = np.arange(0, len(kl_pd.index))
    if showJiJinOldPchangeWave:
        kl_pd['oldPchangeSum'] = kl_pd.p_change.rolling(window=windowBuy).sum()
        kl_pd['oldPchangeSum'].fillna(value=0, inplace=True)
        plt.plot(kl_pd.index, kl_pd.oldPchangeSum, 'o', markersize=3, markeredgewidth=1,
                 markerfacecolor='None', label='p_change2', color='red')
        y_fit = regress_xy_polynomial(x, kl_pd['oldPchangeSum'], poly=poly, zoom=False, show=False)
        plt.plot(kl_pd.index, y_fit)
        plt.title(jiJinCodes[0])
        legendList.append('JiJinOldPchangeWave')
        legendList.append('JiJinOldPchangeWave')

    if showWindowSell:
        plt.plot(kl_pd.index, kl_pd.dayRateSum , 'o', markersize=3, markeredgewidth=1,
                 markerfacecolor='None', label='p_change2', color='blue')  # 基金N日积累量，卖出信号
        y_fit = regress_xy_polynomial(x, kl_pd.dayRateSum, poly=poly, zoom=False, show=False)
        plt.plot(kl_pd.index, y_fit, color='blue')
        legendList.append('WindowSell')
        legendList.append('WindowSell')
    if showWindowBuy:
        plt.plot(kl_pd.index, kl_pd.dayRateSum_old , 'o', markersize=3, markeredgewidth=1,
                 markerfacecolor='None', label='p_change2', color='red')  # 基金N日积累量，买入信号
        y_fit = regress_xy_polynomial(x, kl_pd.dayRateSum_old, poly=poly, zoom=False, show=False)
        plt.plot(kl_pd.index, y_fit)
        y_fit_pd = pd.DataFrame(y_fit,index=kl_pd.index,columns=['y_fit'])
        if keepDays is not None:
            keepDaysPd = y_fit_pd.reindex(keepDays.index)
            plt.plot(keepDaysPd.index, keepDaysPd.y_fit,'o', color='blue')
        legendList.append('WindowBuy')
        legendList.append('WindowBuy')
    plt.legend(legendList,loc="upper left")
    plt.title(str(jiJinCodes) )
    if pltShow :
        plt.show()