from gm.api import set_token, get_instruments, get_history_constituents, get_previous_trading_date

import abupy
from abupy import AbuFactorBuyBreak, AbuFactorAtrNStop, AbuFactorPreAtrNStop, AbuFactorCloseAtrNStop, AbuFactorSellNDay, \
    AbuDownUpTrend, ABuGridHelper, WrsmScorer, GridSearch, abu, AbuMetricsBase, AbuSlippageBuyMean, EStoreAbu
from abupy.BetaBu import ABuPositionBase
from abupy.BetaBu.ABuRatePosition import AbuRatePosition
from abupy.CoreBu import ABuEnv
from abupy.CoreBu.ABuEnv import EDataCacheType, EMarketDataFetchMode, EMarketSourceType, EMarketTargetType
import numpy as np
import tushare as ts
import datetime
from gm.api import *
from pandas.plotting import register_matplotlib_converters #避免画图报警

from abupy.FactorBuyBu.ABuFactorBuyJiJinWave import AbujiJinWave
from abupy.FactorBuyBu.ABuFactorBuyJuBaoPen import AbuJuBaoPen
from abupy.FactorSellBu.ABuFactorSellJiJinWave import AbuFactorSellJiJinWave
from abupy.FactorSellBu.ABuFactorSellJiJinWaveMinMax import AbuFactorSellJiJinWaveMinMax
from abupy.FactorSellBu.ABuFactorSellJuBaoPen import AbuFactorSellJuBaoPen
from abupy.FactorSellBu.ABuFactorSellMinMax import AbuFactorSellMinMax
from abupy.SlippageBu.ABuSlippageSellClose import AbuSlippageSellClose
from abupy.SlippageBu.ABuSlippageSellOpen import AbuSlippageSellOpen
import pandas as pd
from sqlalchemy import  create_engine
register_matplotlib_converters()
#----------------------------资金、股池、基础因子-----------------------------
from abupy.CoreBu.ABuStore import store_abu_result_out_put, store_abu_result_tuple, load_abu_result_tuple, \
    store_python_obj
from abupy.SlippageBu.ABuSlippageBuyOpen import AbuSlippageBuyOpen
connect = create_engine(
    f'mysql+pymysql://dba_4_update:857911Hys@rm-2ze68jb334ufy64rano.mysql.rds.aliyuncs.com:3306/dba_test?charset=utf8')
ABuPositionBase.g_default_pos_class = {'class':AbuRatePosition,'base_rate':0.2} #仓位因子

read_cash = 100000
stock_pickers = None
set_token("8e1026d2dfd455be2e1f239e50004b35a481061e")

data = get_instruments(symbols=None, exchanges=None, sec_types=1, names=None, fields=None, df=True)
stockCode = data[data.trade_n == 0]
# jiJinCode = data[data.sec_name.str.contains('LOF')]\
#     ['sec_id'].array

stockCode = data['sec_id'].array
# choice_symbols = stockCode
choice_symbols = stockCode[0:]
#
# temp = np.arange(0,3500,500)
# temp_month = np.arange(1,5,1)
# symbols = data['symbol'].array[3500:4500]
# for m in temp_month:
#     for i in temp:
#         symbols = data['symbol'].array[i:i+500]
#         # choice_symbols = stockCode
#         date_str='2019-0'+str(m)+'-01'
#         data_jbm =  get_fundamentals(table='trading_derivative_indicator', symbols=symbols, start_date=date_str,
#                                      end_date=date_str, fields='NEGOTIABLEMV,PB,PELFY,TCLOSE,TOTMKTCAP,TRADEDATE,'
#                                                                    'TURNRATE,TOTAL_SHARE,FLOW_SHARE'
#                                      , filter=None, order_by=None, limit=1000, df=True)
#         data_jbm.to_sql("data_jbm"+'_'+date_str, connect, if_exists='append', index=False)
#
# choice_symbols = stockCode
# choice_symbols = ['002718','002719','002721','300740']
# choice_symbols = ['300298']
#
#
# connect = create_engine(
#     f'mysql+pymysql://dba_4_update:857911Hys@rm-2ze68jb334ufy64rano.mysql.rds.aliyuncs.com:3306/dba_test?charset=utf8')
# df = pd.read_sql('select * from orders_pd_source_71',connect)
# choice_symbols = list(set(df['symbol'].array))

# # 获取当前时间
# import time
# # 优化格式化化版本
# now= time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
# # 获取上一个交易日
# last_day = get_previous_trading_date(exchange='SHSE', date=now)
# # 获取沪深300成份股
# stock300 = get_history_constituents(index='SZSE.399008', start_date=last_day,
#                                             end_date=last_day)[0]['constituents'].keys()
# choice_symbols = stock300
ABuEnv.g_cpu_cnt = 8#并发运行线程数
ABuEnv.draw_order_num = 0 #要绘制的订单数
# ABuEnv.g_data_fetch_mode = EMarketDataFetchMode.E_DATA_FETCH_FORCE_NET  #强制从网络获取
ABuEnv.g_data_fetch_mode = EMarketDataFetchMode.E_DATA_FETCH_NORMAL  #强制从网络获取
# ABuEnv.g_data_fetch_mode = EMarketDataFetchMode.E_DATA_FETCH_FORCE_LOCAL  #强制本地，可多线程

ABuEnv.g_market_source = EMarketSourceType.E_MARKET_SOURCE_tx  #作用同上点击效果。腾讯数据源(美股，A股，港股)
ABuEnv.g_market_target = EMarketTargetType.E_MARKET_TARGET_CN
start = datetime.datetime.now()

#----------------------------买入因子-----------------------------
juBaoPenAble = [1]
startDate='2017-03-10'
endDate='2020-04-20'
xd = np.arange(400,410,1000)
skip_days_value = np.arange(10,11,10)
peak_atr = np.arange(0.1,0.2,10)
maxValuePositionStart = np.arange(3,8,100)
maxValuePositionEnd = np.arange(12,13,100)
onlyLastDate = False

params_grid = {'class': [AbuJuBaoPen], 'juBaoPenAble':juBaoPenAble, 'xd':xd,'skip_days_value':skip_days_value,'peak_atr':peak_atr,
               'maxValuePositionStart':[2],
               'maxValuePositionEnd':[1]}

buy_factors_product = ABuGridHelper.gen_factor_grid(ABuGridHelper.K_GEN_FACTOR_PARAMS_BUY, [params_grid])

# abupy.slippage.sbm.g_max_down_rate = 0.03
for buy_factors_product_item in buy_factors_product:
    # buy_factors_product_item[0]['slippage'] = AbuSlippageBuyMinMax #指定买入滑点类
    buy_factors_product_item[0]['slippage'] = AbuSlippageBuyOpen #指定买入滑点类
    # buy_factors_product_item[0]['onlyLastDate'] = onlyLastDate

print('买入因子参数共有{}种组合方式'.format(len(buy_factors_product)))

#----------------------------卖出因子-----------------------------
# sell_n_range = windowBuy
sell_n_range = np.arange(3,4,10) # #设定买入后只持有天数，默认1
stopWinSell = np.arange(0.04,0.06,0.02) # 止盈卖出
stopLoseSell = np.arange(-0.04,-0.02,0.02) # 止损卖出
# not_sell_n_range = np.arange(1,2,1) # #设定买入后只持有天数，默认1
is_sell_today = True #设定买入n天后，当天还是隔天卖出。默认False。
# sell_atr_nstop_factor_grid = {'class':[AbuFactorSellJuBaoPen],'sell_n':sell_n_range,'not_sell_n':not_sell_n_range}
# sell_atr_nstop_factor_grid = {'class':[AbuFactorSellJiJinWaveMinMax],'sell_n':sell_n_range}
# sell_atr_nstop_factor_grid = {'class':[AbuFactorSellJiJinWave]}
sell_atr_nstop_factor_grid = {'class':[AbuFactorSellMinMax],'sell_n':sell_n_range,'stopWinSell':stopWinSell
    ,'stopLoseSell':stopLoseSell}

sell_factors_product = ABuGridHelper.gen_factor_grid(ABuGridHelper.K_GEN_FACTOR_PARAMS_SELL,[sell_atr_nstop_factor_grid])
for sell_factors_product_item in sell_factors_product:
    # sell_factors_product_item[0]['slippage'] = AbuSlippageSellOpen
    sell_factors_product_item[0]['slippage'] = AbuSlippageSellClose
    # sell_factors_product_item[0]['slippage'] = AbuSlippageSellMinMax

print('卖出因子参数共有{}种组合方式'.format(len(sell_factors_product)))
ABuEnv.date_str = datetime.datetime.now().strftime("_%Y_%m_%d")

grid_search = GridSearch(read_cash,choice_symbols,stock_pickers_product=None,buy_factors_product=buy_factors_product,
                         sell_factors_product=sell_factors_product,n_folds=1,start=startDate, end=endDate)
if __name__ == '__main__':    #多线程必须内容，不可删除。
    scores,score_tuple_array = grid_search.fit(n_jobs=-1)
    store_python_obj(score_tuple_array, 'score_tuple_array'+ABuEnv.date_str, show_log=False)
#-------权重方式开始------
    if score_tuple_array != None:
        scorer = WrsmScorer(score_tuple_array, weights=[1, 0, 0, 0])
        scorer_returns_max = scorer.fit_score()
        best_result_tuple = score_tuple_array[scorer_returns_max.index[-1]]
        print(best_result_tuple.buy_factors) #打印最优因子
        print(best_result_tuple.sell_factors) #打印最优因子
        store_abu_result_tuple(best_result_tuple, n_folds=None, store_type=EStoreAbu.E_STORE_CUSTOM_NAME,
                               custom_name='best_1_0_0_0')
        AbuMetricsBase.show_general(best_result_tuple.orders_pd, best_result_tuple.action_pd,
                                    best_result_tuple.capital, best_result_tuple.benchmark, only_info=False)

        scorer = WrsmScorer(score_tuple_array, weights=[0, 1, 0, 0])
        scorer_returns_max = scorer.fit_score()
        best_result_tuple = score_tuple_array[scorer_returns_max.index[-1]]
        print(best_result_tuple.buy_factors)  # 打印最优因子
        print(best_result_tuple.sell_factors)  # 打印最优因子
        store_abu_result_tuple(best_result_tuple, n_folds=None, store_type=EStoreAbu.E_STORE_CUSTOM_NAME,
                               custom_name='best_0_1_0_0')
        AbuMetricsBase.show_general(best_result_tuple.orders_pd, best_result_tuple.action_pd,
                                    best_result_tuple.capital, best_result_tuple.benchmark, only_info=True)
    else:
        print("没有订单生成！")
    end = datetime.datetime.now()
    print("程序运行时间：" +str(end - start))
#-------权重方式结束------

# abu_result_tuple,kl_pd_manager = abu.run_loop_back(read_cash,buy_factors,sell_factors,stock_pickers,choice_symbols=choice_symbols,n_folds = 1)
#
# from abupy import AbuMetricsBase
# metrics =  AbuMetricsBase(*abu_result_tuple)
# metrics.fit_metrics()
# metrics.plot_returns_cmp()#收益对照，收益线性拟合，资金概率密度
# metrics.plot_sharp_volatility_cmp() #策略与基准之间波动率和夏普比率
# metrics.plot_effect_mean_day()#买入因子平均生效间隔
# metrics.plot_keep_days() #策略持股天数
# metrics.plot_sell_factors()#策略卖出因子生效分布情况
# metrics.plot_max_draw_down()#最大回撤