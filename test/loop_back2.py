from abupy import AbuFactorBuyBreak, AbuFactorAtrNStop, AbuFactorPreAtrNStop, AbuFactorCloseAtrNStop, AbuBenchmark, \
    ABuRegUtil
from abupy import  abu,AbuMetricsBase
from abupy.CoreBu import ABuEnv
from abupy.CoreBu.ABuEnv import EDataCacheType, EMarketDataFetchMode, EMarketSourceType ,EMarketTargetType
import numpy as np
import matplotlib.pyplot as plt
from abupy.CoreBu.ABuStore import store_abu_result_out_put

ABuEnv.g_cpu_cnt = 1 #并发运行线程数
ABuEnv.draw_order_num = 0 #要绘制的订单数
#ABuEnv.g_data_fetch_mode = EMarketDataFetchMode.E_DATA_FETCH_NORMAL  #本地结合网络获取
ABuEnv.g_data_fetch_mode = EMarketDataFetchMode.E_DATA_FETCH_FORCE_NET  #强制从网络获取
# widget.tt.date_source = EMarketSourceType.E_MARKET_SOURCE_tx.value #腾讯数据源(美股，A股，港股)
ABuEnv.g_market_source = EMarketSourceType.E_MARKET_SOURCE_tx  #作用同上点击效果。百度数据源(美股，A股，港股)
ABuEnv.g_market_target = EMarketTargetType.E_MARKET_TARGET_CN
read_cash = 100000
stock_pickers = None
buy_factors = [{'xd':60,'class':AbuFactorBuyBreak},{'xd':42,'class':AbuFactorBuyBreak}]
sell_factors = [{'stop_loss_n':1.0,'stop_win_n':3.0,'class':AbuFactorAtrNStop}]

industry_data = ts.get_industry_classified() #获取行业信息
industry_data = industry_data[~industry_data.name.str.contains('ST')]
industry_data2 = industry_data['code'][industry_data.c_name == '公路桥梁'] #过滤
# industry_data2 = industry_data['code']
choice_symbols2 = industry_data2.values
choice_symbols = choice_symbols2

# choice_symbols = ['000895']



benchmark = AbuBenchmark(n_folds=1, start=None, end=None)
kl_pd = benchmark.kl_pd

pd_len = kl_pd.shape[0]
kl_pd = kl_pd[kl_pd.key > pd_len-16][kl_pd.key < pd_len-11]

plt.plot(kl_pd.index,kl_pd.close)
plt.show()

ang = ABuRegUtil.calc_regress_deg(kl_pd.close, show=True)
print('test OK')
# factor_dict = {'class': AbuPickRegressAngMinMax,
#                        'xd': self.xd.value,
#                        'reversed': self.reversed.value,
#                        'threshold_ang_min': ang_min,
#                        'threshold_ang_max': ang_max}
#
# factor_desc_key = u'角度选股最大:{}最小:{},周期:{},反转:{}'.format(
#     ang_max, ang_min, self.xd.value, self.reversed.value)

# abu_result_tuple,kl_pd_manager = abu.run_loop_back(read_cash,buy_factors,sell_factors,stock_pickers,choice_symbols=choice_symbols,n_folds = 1)
# store_abu_result_out_put(abu_result_tuple)
