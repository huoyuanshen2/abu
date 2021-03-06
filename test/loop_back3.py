from gm.api import set_token, get_instruments

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
from pandas.plotting import register_matplotlib_converters #避免画图报警

from abupy.FactorBuyBu.ABuFactorBuyJiJinWave import AbujiJinWave
from abupy.FactorSellBu.ABuFactorSellJiJinWave import AbuFactorSellJiJinWave
from abupy.FactorSellBu.ABuFactorSellMinMax import AbuFactorSellMinMax
from abupy.SlippageBu.ABuSlippageBuyMinMax import AbuSlippageBuyMinMax
from abupy.SlippageBu.ABuSlippageSellClose import AbuSlippageSellClose
from abupy.SlippageBu.ABuSlippageSellMinMax import AbuSlippageSellMinMax

register_matplotlib_converters()
#----------------------------资金、股池、基础因子-----------------------------
from abupy.CoreBu.ABuStore import store_abu_result_out_put, store_abu_result_tuple, load_abu_result_tuple, \
    store_python_obj
from abupy.FactorBuyBu.ABuFactorBuyBigNDown import AbubigNDown
from abupy.FactorBuyBu.ABuFactorBuyBigWave2 import AbuBigWave2
from abupy.FactorBuyBu.ABuFactorBuyIndustry import AbuIndustryRateBuy
from abupy.SlippageBu.ABuSlippageBuyOpen import AbuSlippageBuyOpen
from abupy.SlippageBu.ABuSlippageSellOpen import AbuSlippageSellOpen


ABuPositionBase.g_default_pos_class = {'class':AbuRatePosition,'base_rate':0.2} #仓位因子

read_cash = 100000
stock_pickers = None
industry_data = ts.get_industry_classified() #获取行业信息
industry_data = industry_data[~industry_data.name.str.contains('ST')]
# industry_data2 = industry_data['code'][industry_data.c_name == '公路桥梁'] #过滤
# industry_data2 = industry_data['code'][industry_data.code == '600150']
# choice_symbols2 = industry_data2.values
# choice_symbols2 = ['501002','501005','501009','501010']
# choice_symbols2 = ['501002','501005']
set_token("8e1026d2dfd455be2e1f239e50004b35a481061e")
# data = get_instrumentinfos(symbols=None, exchanges=None, sec_types=2, names=None, fields=None, df=True)
data = get_instruments(symbols=None, exchanges=None, sec_types=2, names=None, fields=None, df=True)
jiJinCode = data[data.trade_n == 0]
jiJinCode = data[data.sec_name.str.contains('LOF')]\
    ['sec_id'].array
# choice_symbols = jiJinCode
# choice_symbols = ['160809']

ABuEnv.g_cpu_cnt = 4#并发运行线程数
ABuEnv.draw_order_num = 0 #要绘制的订单数
ABuEnv.g_data_fetch_mode = EMarketDataFetchMode.E_DATA_FETCH_FORCE_NET  #强制从网络获取
# ABuEnv.g_data_fetch_mode = EMarketDataFetchMode.E_DATA_FETCH_FORCE_LOCAL  #强制本地，可多线程

ABuEnv.g_market_source = EMarketSourceType.E_MARKET_SOURCE_tx  #作用同上点击效果。腾讯数据源(美股，A股，港股)
ABuEnv.g_market_target = EMarketTargetType.E_MARKET_TARGET_CN
start = datetime.datetime.now()

#----------------------------买入因子-----------------------------
maxChangeRateAble = [1]
maxChangeRate_value_range = np.arange(9.6,10.5,1) #最大涨幅比例

targetCountRateAble = [0]
changeRate_value_range = np.arange(-4,-3,1) #涨幅比例
targetCountRate_value_range = np.arange(1,2,1) # 参考相关度高的股票数量

todayGT0DataAble = [0]
targetGT0CountRate_value_range = np.arange(90,100,10) # 所属行业有多少比例涨幅》0时，算作有效买入条件之一。

maxPriceRateAble = [0] #当天详细数据最高点占比
maxPriceRate_value_range = np.arange(80,90,10)

xdAble_value = [0]
xd_value_range = np.arange(20,30,10)

tradingDaysAble_value = [1]
tradingDays_range = np.arange(1,3,1)
targetGTX_range = np.arange(5,8,1)
targetGTXRate_range = np.arange(5,8,1)

buy_bk_factor_grid = {'class': [AbuIndustryRateBuy], 'maxPriceRateAble':maxPriceRateAble,'targetCountRateAble':targetCountRateAble,
                      'maxChangeRateAble':maxChangeRateAble,'todayGT0DataAble':todayGT0DataAble,
                      'changeRate': changeRate_value_range,'maxChangeRate': maxChangeRate_value_range,
                       'targetCountRate': targetCountRate_value_range,'targetGT0CountRate': targetGT0CountRate_value_range,
                      'maxPriceRate': maxPriceRate_value_range,'xd': xd_value_range,'xdAble': xdAble_value,
                      'tradingDaysAble': tradingDaysAble_value,'tradingDays': tradingDays_range,'targetGTX': targetGTX_range,
                      'targetGTXRate': targetGTXRate_range}

bigWave2Able = [1]
bigWave = np.arange(0.1,0.2,0.1)
xd_value_range = np.arange(20,30,10)
AbuBigWave2_grid = {'class': [AbuBigWave2], 'bigWave2Able':bigWave2Able,'bigWave':bigWave,'xd':xd_value_range}

bigNDownAble = [1]
downDays = np.arange(2,3,1)
downRate = np.arange(-3,-2,1)
xd_value_range = np.arange(10,20,10)
openJClose = np.arange(0.02,0.03,0.01)
benchXdAble = [1]
benchXd = np.arange(5,6,1)
targetAng = np.arange(-150,-140,10)
AbuBigNDown_grid = {'class': [AbubigNDown], 'bigNDownAble':bigNDownAble,'downDays':downDays,'downRate':downRate,
                    'openJClose':openJClose,'xd':xd_value_range,'benchXdAble':benchXdAble,
                    'benchXd':benchXd,'targetAng':targetAng}
#---------------------------------基金周期因子-------------------------
jiJinWaveAble = [1]
windowBuy = np.arange(18,19,1)
windowSell = np.arange(10,11,1)
poly = np.arange(50,60,10)
pears_value = np.arange(0.95,0.96,0.1)
targetUpAng = np.arange(200,310,1000)
targetDownAng = np.arange(-30,-20,3000)
skip_days_value = [0]
xd_value_range = np.arange(200,201,1)
stopWinSellPriceRate = [0.08]
stopLoseSellPriceRate = [-0.08]
waveMax = [8]
waveMin = [-8]
startDate='2019-05-09'
endDate='2020-03-06'
jiJinWave_grid = {'class': [AbujiJinWave], 'jiJinWaveAble':jiJinWaveAble,'windowBuy':windowBuy,'windowSell':windowSell,
                    'poly':poly,'xd':xd_value_range,'skip_days_value':skip_days_value,'pears_value':pears_value,
                    'targetUpAng':targetUpAng,'targetDownAng':targetDownAng,
                  'stopWinSellPriceRate':stopWinSellPriceRate,'stopLoseSellPriceRate':stopLoseSellPriceRate,'waveMax':waveMax,'waveMin':waveMin}

buy_factors_product = ABuGridHelper.gen_factor_grid(ABuGridHelper.K_GEN_FACTOR_PARAMS_BUY,[jiJinWave_grid])

# abupy.slippage.sbm.g_max_down_rate = 0.03
for buy_factors_product_item in buy_factors_product:
    # buy_factors_product_item[0]['slippage'] = AbuSlippageBuyMinMax #指定买入滑点类
    buy_factors_product_item[0]['slippage'] = AbuSlippageBuyOpen #指定买入滑点类

print('买入因子参数共有{}种组合方式'.format(len(buy_factors_product)))

#----------------------------卖出因子-----------------------------
sell_n_range = windowBuy
# sell_n_range = np.arange(1,2,1) # #设定买入后只持有天数，默认1
is_sell_today = True #设定买入n天后，当天还是隔天卖出。默认False。
# sell_atr_nstop_factor_grid = {'class':[AbuFactorSellNDay],'sell_n':sell_n_range}
sell_atr_nstop_factor_grid = {'class':[AbuFactorSellJiJinWave]}
# sell_atr_nstop_factor_grid = {'class':[AbuFactorSellMinMax]}

sell_factors_product = ABuGridHelper.gen_factor_grid(ABuGridHelper.K_GEN_FACTOR_PARAMS_SELL,[sell_atr_nstop_factor_grid])
for sell_factors_product_item in sell_factors_product:
    # sell_factors_product_item[0]['slippage'] = AbuSlippageSellOpen
    sell_factors_product_item[0]['slippage'] = AbuSlippageSellClose
    # sell_factors_product_item[0]['slippage'] = AbuSlippageSellMinMax

print('卖出因子参数共有{}种组合方式'.format(len(sell_factors_product)))
ABuEnv.date_str = datetime.datetime.now().strftime("_%Y_%m_%d")

grid_search = GridSearch(read_cash,choice_symbols,buy_factors_product=buy_factors_product,
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