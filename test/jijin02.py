from gm.api import set_token, get_instrumentinfos, get_instruments

from abupy import EStoreAbu, IndexSymbol, ABuSymbolPd, EMarketDataSplitMode, EMarketDataFetchMode, EMarketSourceType, \
    EMarketTargetType, ParameterGrid, GridSearch
from abupy.CoreBu import ABuEnv
import numpy as np
from abupy.MetricsBu import ABuGridHelper

# ABuEnv.g_data_fetch_mode = EMarketDataFetchMode.E_DATA_FETCH_NORMAL  #强制本地，可多线程
ABuEnv.g_data_fetch_mode = EMarketDataFetchMode.E_DATA_FETCH_FORCE_NET  #强制本地，可多线程

ABuEnv.g_market_source = EMarketSourceType.E_MARKET_SOURCE_tx  #作用同上点击效果。腾讯数据源(美股，A股，港股)
ABuEnv.g_market_target = EMarketTargetType.E_MARKET_TARGET_CN
# jiJinCode = ['003095','002844','005311','006265','001195','006109','398041','003834'] #基金数据

set_token("8e1026d2dfd455be2e1f239e50004b35a481061e")
# data = get_instrumentinfos(symbols=None, exchanges=None, sec_types=2, names=None, fields=None, df=True)
data = get_instruments(symbols=None, exchanges=None, sec_types=2, names=None, fields=None, df=True)
jiJinCode = data[data.trade_n == 0]
# jiJinCode = data[data.sec_name.str.contains('LOF')]\
#     ['sec_id'].head(10).array

# jiJinCode = data[data.sec_id == '501002']
# jiJinCode = ['501005','501009','501010'] #基金数据
jiJinCode = ['501009'] #基金数据
jiJinCode2 = ['501005','501015','501016','501017','501019','501021','501029','501032','501050','501057','501062','501067',
 '501072','501086','501106','501186','501188','501189','501300','501301','501310','160106','160119','160130',
 '160133','160140','160142','160211','160212','160215','160216','160223','160225','160311','160314','160323',
 '160421','160505','160515','160518','160607','160611','160618','160635','160706','160716','160717','160719',
 '160723','160812','160813','160915','160916','160918','160921','161005','161010','161017','161019','161033',
 '161035','161115','161116','161118','161119','161125','161126','161127','161128','161129','161130','161131',
 '161132','161216','161225','161226','161227','161232','161607','161631','161706','161713','161716','161728',
 '161810','161815','161820','161834','161903','162006','162108','162207','162411','162414','162415','162605',
 '162607','162703','162711','162712','162719','163001','163110','163111','163402','163407','163412','163415',
 '163417','163503','163801','163819','164210','164701','164808','164824','164902','164906','165513','165525',
 '165528','166001','166008','166016','167501','168104','168106','169101','169105','169301']
# jiJinCode = jiJinCode2[0:800]
startDate = ['2019-03-03']
endDate = ['2020-01-02']
windowBuy = np.arange(12,13,2)
windowSell = np.arange(18,19,1)
poly = np.arange(50,60,10)
showWindowBuy=True
showWindowSell=False
showMutil=False
AbuBigNDown_grid2 = {'jiJinCode':['123456'],'startDate': startDate, 'endDate':endDate,'windowBuy':windowBuy,'windowSell':windowSell,
                    'poly':poly}

buy_factors_product = ABuGridHelper.gen_factor_grid_4_jijin([AbuBigNDown_grid2])

print('买入因子参数共有{}种组合方式'.format(len(buy_factors_product)))
grid_search = GridSearch(None,jiJinCode,buy_factors_product=buy_factors_product,
                         sell_factors_product=None,n_folds=1,start=startDate[0], end=endDate[0])
if __name__ == '__main__':    #多线程必须内容，不可删除。
    keysObj = {'showMutil':showMutil}
    scores,score_tuple_array = grid_search.fit4JiJin(n_jobs=-1,showWindowBuy=showWindowBuy,showWindowSell=showWindowSell,
                                                     keysObj=keysObj)
