from abupy import AbuFactorBuyBreak, AbuFactorAtrNStop, AbuFactorPreAtrNStop, AbuFactorCloseAtrNStop, AbuFactorSellNDay, \
    AbuDownUpTrend, ABuGridHelper, WrsmScorer, GridSearch, abu, AbuMetricsBase, AbuSlippageBuyMean, AbuBenchmark, \
    AbuKLManager, AbuCapital
from abupy.BetaBu import ABuPositionBase
from abupy.BetaBu.ABuRatePosition import AbuRatePosition
from abupy.CoreBu import ABuEnv
from abupy.CoreBu.ABuEnv import EDataCacheType, EMarketDataFetchMode, EMarketSourceType, EMarketTargetType
import numpy as np
import tushare as ts
import pandas as pd

#----------------------------资金、股池、基础因子-----------------------------
from abupy.SlippageBu.ABuSlippageBuyOpen import AbuSlippageBuyOpen
from abupy.SlippageBu.ABuSlippageSellOpen import AbuSlippageSellOpen

changeRate  = -2 #一个行业有效涨幅百分比例
targetCountRate = 0.3 #一个行业中，有多少比例符合涨幅要求算通过


industry_data = ts.get_industry_classified() #获取行业信息
# industry_data = industry_data[~industry_data.name.str.contains('ST')] #过滤st数据。
distinct_industrys = industry_data.drop_duplicates(['c_name'], keep='last').c_name[0:2]
distinct_industrys = distinct_industrys.values.tolist()
industry_data3 = industry_data[industry_data.c_name.isin(distinct_industrys)]

industry_data3 = industry_data3[~industry_data3.name.str.contains('ST')] #过滤st数据。

industry_data4 = industry_data3['code']
industry_data4 = industry_data4.drop_duplicates()
# industry_data2
choice_symbols = industry_data4.values[0:5] #这里是股票池参数
print(choice_symbols)

# for stock in choice_symbols :
#     stock_kd = pd.getByName(stock)


ABuEnv.g_cpu_cnt = 4#并发运行线程数
ABuEnv.draw_order_num = 0 #要绘制的订单数
ABuEnv.g_data_fetch_mode = EMarketDataFetchMode.E_DATA_FETCH_FORCE_NET  #强制从网络获取
#ABuEnv.g_data_fetch_mode = EMarketDataFetchMode.E_DATA_FETCH_FORCE_LOCAL  #强制本地，可多线程

ABuEnv.g_market_source = EMarketSourceType.E_MARKET_SOURCE_tx  #作用同上点击效果。腾讯数据源(美股，A股，港股)
ABuEnv.g_market_target = EMarketTargetType.E_MARKET_TARGET_CN

benchmark = AbuBenchmark(n_folds=1, start=None, end=None)
# 资金类初始化
capital = AbuCapital(1000000, benchmark, user_commission_dict=None)
# kl数据管理类初始化
kl_pd_manager = AbuKLManager(benchmark, capital)
# 批量获取择时kl数据
industryPchangeAllDaetDic = dict()
if __name__ == '__main__':
    kl_pd_manager.batch_get_pick_time_kl_pd(choice_symbols, n_process=4)
    kl_pd_dick = kl_pd_manager.pick_kl_pd_dict['pick_time']
    for stockCode, stockData in kl_pd_dick.items():
        if stockData is None:
            continue
        pd_data = stockData.p_change
        stockIndustryNames = industry_data['c_name'][industry_data.code == stockCode]
        for _, stockIndustryName in stockIndustryNames.items():
            if stockIndustryName in industryPchangeAllDaetDic.keys():
                left = industryPchangeAllDaetDic[stockIndustryName]
                right = pd.DataFrame({'date': stockData.date, stockCode: stockData.p_change})
                industryPchangeAllDaetDic[stockIndustryName] = pd.merge(left, right, on='date', how='outer')
            else :
                left = pd.DataFrame({'date': stockData.date, stockCode: stockData.p_change})
                industryPchangeAllDaetDic[stockIndustryName] = left

    temp = industryPchangeAllDaetDic['综合行业'].date.head(1)
    # print(industryPchangeAllDaetDic['综合行业'].date.head(1))
    print(temp.values)
    print(type(temp.values[0]))
    targetStock = '600051'
    testDate = 20190222
    stockIndustryNames = industry_data['c_name'][industry_data.code == stockCode]
    for _, stockIndustryName in stockIndustryNames.items():
        industryData = industryPchangeAllDaetDic[stockIndustryName]
        industryTodayData = industryData[industryData.date == testDate][0:1]
        industryTodayData.drop('date',axis=1,  inplace=True)
        df2 = pd.DataFrame(industryTodayData.values.T, index=industryTodayData.columns, columns=['p_change'])
        print(df2)
        df3 = df2[df2.p_change > changeRate ]
        allCount = df2.shape[0]
        ableCount = df3.shape[0]
        if allCount > 0:
            countRate = allCount/ableCount
            if countRate < targetCountRate :
                continue
        if df3.index.contains(targetStock):
            print("买入代码:"+targetStock)
            print("买入代码涨幅:" + str(df3['p_change'][df3.index == targetStock]))
            print("发出买入信号")


print("OK!")
