from abupy import AbuFactorBuyBreak, AbuFactorAtrNStop, AbuFactorPreAtrNStop, AbuFactorCloseAtrNStop, AbuFactorSellNDay, \
    AbuDownUpTrend, ABuGridHelper, WrsmScorer, GridSearch, abu, AbuMetricsBase, AbuSlippageBuyMean, EStoreAbu, \
    code_to_symbol, AbuBenchmark
from abupy.BetaBu import ABuPositionBase
from abupy.BetaBu.ABuRatePosition import AbuRatePosition
from abupy.CoreBu import ABuEnv
from abupy.CoreBu.ABuEnv import EDataCacheType, EMarketDataFetchMode, EMarketSourceType, EMarketTargetType
import numpy as np
import tushare as ts
import datetime
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import pandas as pd
from sqlalchemy import create_engine

connect = create_engine(
    f'mysql+pymysql://dba_4_update:857911Hys@rm-2ze68jb334ufy64rano.mysql.rds.aliyuncs.com:3306/dba_test?charset=utf8')


#----------------------------资金、股池、基础因子-----------------------------
from abupy.CoreBu.ABuStore import store_abu_result_out_put, store_abu_result_tuple, load_abu_result_tuple, \
    store_python_obj, load_python_obj
from abupy.MarketBu.ABuDataCache import load_kline_df
from abupy.TradeBu.ABuTradeDrawer import plot_result_price, plot_result_volumn, plot_result_volumn_sum, \
    plot_result_pearsonr, plot_order_myself
from abupy.UtilBu.AbuIndustryDataUtil import getIndustryDataLocal

ABuEnv.g_data_fetch_mode = EMarketDataFetchMode.E_DATA_FETCH_FORCE_LOCAL  #强制本地，可多线程

ABuEnv.g_market_source = EMarketSourceType.E_MARKET_SOURCE_tx  #作用同上点击效果。腾讯数据源(美股，A股，港股)
ABuEnv.g_market_target = EMarketTargetType.E_MARKET_TARGET_CN

start = datetime.datetime.now()
# ABuEnv.date_str = datetime.datetime.now().strftime("_%Y_%m_%d")
ABuEnv.date_str = "_2020_04_26"

# getIndustryDataLocal()
score_tuple_array = load_python_obj('score_tuple_array'+ABuEnv.date_str)
#-------权重方式开始------
if score_tuple_array != None:
    scorer = WrsmScorer(score_tuple_array, weights=[1, 0, 0, 0])
    test = scorer.score_pd
    scorer_returns_max = scorer.fit_score()
    best_result_tuple = score_tuple_array[scorer_returns_max.index[-1]]
    orders_pd = best_result_tuple.orders_pd[0:100]
    # 如果想要自动建表的话把if_exists的值换为replace/append, 建议自己建表
    orders_pd.to_sql("orders_pd"+ABuEnv.date_str, connect, if_exists='replace', index=False)
    key_value = 1000 * 10000

    for key,order in orders_pd.iterrows() :
        if order.sell_type != 'keep':
            plot_order_myself(order)

    print(best_result_tuple.buy_factors) #打印最优因子
    print(best_result_tuple.sell_factors) #打印最优因子
    store_abu_result_tuple(best_result_tuple, n_folds=None, store_type=EStoreAbu.E_STORE_CUSTOM_NAME,
                           custom_name='best_1_0_0_0')

    AbuMetricsBase.show_general(best_result_tuple.orders_pd, best_result_tuple.action_pd,
                                best_result_tuple.capital, best_result_tuple.benchmark, only_info=True)
else:
    print("没有订单生成！")
end = datetime.datetime.now()
print("程序运行时间：" +str(end - start))
