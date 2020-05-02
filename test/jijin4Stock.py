from abupy import EStoreAbu, IndexSymbol, ABuSymbolPd, EMarketDataSplitMode, EMarketDataFetchMode, EMarketSourceType, \
    EMarketTargetType, ParameterGrid, GridSearch
from abupy.CoreBu import ABuEnv
import numpy as np
from abupy.MetricsBu import ABuGridHelper

ABuEnv.g_data_fetch_mode = EMarketDataFetchMode.E_DATA_FETCH_NORMAL
# ABuEnv.g_data_fetch_mode = EMarketDataFetchMode.E_DATA_FETCH_FORCE_NET

ABuEnv.g_market_source = EMarketSourceType.E_MARKET_SOURCE_tx  #作用同上点击效果。腾讯数据源(美股，A股，港股)
ABuEnv.g_market_target = EMarketTargetType.E_MARKET_TARGET_CN
# symbols_dic={'行业':'黄金','数据':['600489','002155','600547','601069','601899']} #更新后需要手动删除历史数据
symbols_dic={'行业':'黄金','数据':['600489']} #更新后需要手动删除历史数据
# symbols_dic={'行业':'中欧医疗','数据':['300015','000661','600276','300347','603259']} #更新后需要手动删除历史数据
# symbols_dic={'行业':'铁矿石','数据':['600808','600581','000898','600307','600117']} #更新后需要手动删除历史数据
# symbols_dic={'行业':'水产养殖','数据':['300094','600258','600467','002069']} #更新后需要手动删除历史数据
# symbols_dic={'行业':'养殖业','数据':['300094','002458','002321','600257','000735']} #更新后需要手动删除历史数据
windowBuy = np.arange(7,15,1)#可单独显示
poly = np.arange(50,60,10)
showWindowBuy=True
showWindowSell=True
startDate = ['2019-01-01'] #更新后需要手动删除历史数据
endDate = ['2020-02-26']
concept=symbols_dic['行业']
symbols=symbols_dic['数据']
windowSell = np.arange(7,8,1)
AbuBigNDown_grid2 = {'startDate': startDate, 'endDate':endDate,'windowBuy':windowBuy,'windowSell':windowSell,
                    'poly':poly}

buy_factors_product = ABuGridHelper.gen_factor_grid_4_jijin([AbuBigNDown_grid2])

print('买入因子参数共有{}种组合方式'.format(len(buy_factors_product)))
grid_search = GridSearch(100000,choice_symbols=symbols,buy_factors_product=buy_factors_product,
                         sell_factors_product=None,start=startDate[0], end=endDate[0],concept=concept)
if __name__ == '__main__':    #多线程必须内容，不可删除。
    scores,score_tuple_array = grid_search.fit4JiJin(n_jobs=1,isJiJin=False,showWindowBuy=showWindowBuy,showWindowSell=showWindowSell,concept=concept)
