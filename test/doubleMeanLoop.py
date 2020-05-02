from gm.api import set_token, get_instruments
from abupy import AbuFactorBuyBreak, AbuFactorAtrNStop, AbuFactorPreAtrNStop, AbuFactorCloseAtrNStop, AbuFactorSellNDay, \
    AbuDownUpTrend, ABuGridHelper, WrsmScorer, GridSearch, abu, AbuMetricsBase, AbuSlippageBuyMean, EStoreAbu, \
    AbuDoubleMaBuy, AbuDoubleMaSell
from abupy.BetaBu import ABuPositionBase
from abupy.BetaBu.ABuRatePosition import AbuRatePosition
from abupy.CoreBu import ABuEnv
from abupy.CoreBu.ABuEnv import EDataCacheType, EMarketDataFetchMode, EMarketSourceType, EMarketTargetType
import numpy as np
import tushare as ts
import datetime
from pandas.plotting import register_matplotlib_converters #避免画图报警

from abupy.FactorBuyBu.ABuFactorBuyDMPlus import AbuDoubleMaPlusBuy
from abupy.SlippageBu.ABuSlippageSellClose import AbuSlippageSellClose
register_matplotlib_converters()
#----------------------------资金、股池、基础因子-----------------------------
from abupy.CoreBu.ABuStore import store_abu_result_out_put, store_abu_result_tuple, load_abu_result_tuple, \
    store_python_obj
from abupy.SlippageBu.ABuSlippageBuyOpen import AbuSlippageBuyOpen

ABuPositionBase.g_default_pos_class = {'class':AbuRatePosition,'base_rate':0.2} #仓位因子

read_cash = 100000
stock_pickers = None
industry_data = ts.get_industry_classified() #获取行业信息
industry_data = industry_data[~industry_data.name.str.contains('ST')]
set_token("8e1026d2dfd455be2e1f239e50004b35a481061e")
data = get_instruments(symbols='SHSE.000016', exchanges=None, sec_types=None, names=None, fields=None, df=True)
# jiJinCode = data[data.trade_n == 0]
jiJinCode = data['sec_id'].array
# choice_symbols = jiJinCode[0:3]
# choice_symbols = ['000895']
choice_symbols = ['159901']
ABuEnv.g_cpu_cnt = 4#并发运行线程数
ABuEnv.draw_order_num = 0 #要绘制的订单数
ABuEnv.g_data_fetch_mode = EMarketDataFetchMode.E_DATA_FETCH_FORCE_NET  #强制从网络获取
# ABuEnv.g_data_fetch_mode = EMarketDataFetchMode.E_DATA_FETCH_FORCE_LOCAL  #强制本地，可多线程

ABuEnv.g_market_source = EMarketSourceType.E_MARKET_SOURCE_tx  #作用同上点击效果。腾讯数据源(美股，A股，港股)
ABuEnv.g_market_target = EMarketTargetType.E_MARKET_TARGET_CN
start = datetime.datetime.now()

#----------------------------买入因子-----------------------------


fast = np.arange(25,30,5)
slow = np.arange(40,41,5)
buy_bk_factor_grid = {'class': [AbuDoubleMaPlusBuy], 'fast':fast,'slow':slow}
startDate='2015-03-01'
endDate='2020-03-15'
buy_factors_product = ABuGridHelper.gen_factor_grid(ABuGridHelper.K_GEN_FACTOR_PARAMS_BUY,[buy_bk_factor_grid])

for buy_factors_product_item in buy_factors_product:
    buy_factors_product_item[0]['slippage'] = AbuSlippageBuyOpen #指定买入滑点类

print('买入因子参数共有{}种组合方式'.format(len(buy_factors_product)))

#----------------------------卖出因子-----------------------------
sell_fast = np.arange(20,30,5)
sell_slow = np.arange(50,62,5)
sell_atr_nstop_factor_grid = {'class':[AbuDoubleMaSell], 'fast':sell_fast,'slow':sell_slow}

sell_factors_product = ABuGridHelper.gen_factor_grid(ABuGridHelper.K_GEN_FACTOR_PARAMS_SELL,[sell_atr_nstop_factor_grid])
for sell_factors_product_item in sell_factors_product:
    sell_factors_product_item[0]['slippage'] = AbuSlippageSellClose

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