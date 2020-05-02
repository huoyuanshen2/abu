from abupy import AbuBenchmark, EMarketDataFetchMode, EMarketSourceType

# widget.tt.store_mode = EDataCacheType.E_DATA_CACHE_CSV.value  #缓存模式 csv模式(推荐)
from abupy.CoreBu import ABuEnv
from  abupy import ABuSymbolPd
import pandas as pd
import matplotlib.pyplot as plt
ABuEnv.g_data_fetch_mode = EMarketDataFetchMode.E_DATA_FETCH_NORMAL  #本地结合网络获取
#ABuEnv.g_data_fetch_mode = EMarketDataFetchMode.E_DATA_FETCH_FORCE_NET  #强制从网络获取
ABuEnv.g_market_source = EMarketSourceType.E_MARKET_SOURCE_tx  #作用同上点击效果。百度数据源(美股，A股，港股)

# -------------------------------数据获取------------------
# 获取指定年份数据
# 方法1
benchmark = AbuBenchmark(benchmark='sz000002', start=None, end=None, n_folds=1, rs=True, benchmark_kl_pd=None)
# benchmark = AbuBenchmark(benchmark='sz000002', start='2018-02-02', end='2019-04-02', n_folds=None, rs=True, benchmark_kl_pd=None)
# print(benchmark.kl_pd)
# 方法2
# tsla_df = ABuSymbolPd.make_kl_df('sz000002',n_folds=2)
# print(tsla_df.tail())
# # 这里要把类型转换为time
# df.index = pd.to_datetime(df.index)
# df['key'] = list(range(0, len(df)))
# -----------------------------------基本画图---------------------
#收盘价图
def plot_demo(axs=None ,just_series=True):
    drawer = plt if axs is None else axs
    drawer.plot(benchmark.kl_pd.close,c='r')
    plt.xlabel('time')
    plt.ylabel('close')
    plt.title('wan ke A')
    plt.grid(True)
    plt.show()
# plot_demo()

#------------------------------价格+均线-----------------
# benchmark.kl_pd.close.plot()
# from abupy import pd_rolling_std, pd_ewm_std, pd_rolling_mean
# pd_rolling_mean(benchmark.kl_pd.close, window=5).plot()
# pd_rolling_mean(benchmark.kl_pd.close, window=30).plot()
# plt.legend(['close','5 mv','30 mv'],loc='best')
# plt.show()



# 蜡烛图_非交互
from abupy import ABuMarketDrawing
ABuMarketDrawing.plot_candle_form_klpd(benchmark.kl_pd[:90],html_bk=False)
# 蜡烛图_交互
# ABuMarketDrawing.plot_candle_form_klpd(benchmark.kl_pd,html_bk=True)

# -------------------------------数据过滤---------------------
import numpy as np
tsla_df = benchmark.kl_pd

# print(list(tsla_df)) #打印列名
# print(tsla_df.head()) #前5行

# key1 = tsla_df.shape[0] #矩阵第一维长度
# # print(key1)
# --------低开高收数据过滤--------
# low_to_high_df = tsla_df.iloc[tsla_df[(tsla_df.close > tsla_df.open) & (tsla_df.key != tsla_df.shape[0] - 1)].key.values + 1]
# change_ceil_floor = np.where(low_to_high_df['p_change'] > 0, np.ceil(low_to_high_df['p_change']), np.floor(low_to_high_df['p_change']))
# # print(change_ceil_floor)
# change_ceil_floor = pd.Series(change_ceil_floor)  #构建一个单列的序列
# # print(change_ceil_floor)
# print('低开高收的下一个交易日所有下跌的跌幅取整和sum: ' + str(
#         change_ceil_floor[change_ceil_floor < 0].sum()))
#
# print('低开高收的下一个交易日所有上涨的涨幅取整和sum: ' + str(
#         change_ceil_floor[change_ceil_floor > 0].sum()))

# --------数据高涨幅过滤--------
high_p_change_df = tsla_df.iloc[tsla_df[(tsla_df.p_change > 5) & (tsla_df.key != tsla_df.shape[0] - 1)].key.values] #涨幅大于5%的数据
print(high_p_change_df)
print(high_p_change_df.shape[0])
high_p_change_df_1 = tsla_df.iloc[tsla_df[(tsla_df.p_change > 5) & (tsla_df.key != tsla_df.shape[0] - 1)].key.values+1] #涨幅大于5%的下一天数据
print(high_p_change_df_1)










