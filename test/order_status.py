import datetime

import tushare as ts
import pandas as pd
import matplotlib.pyplot as plt

from abupy.CoreBu import ABuEnv
from abupy.CoreBu.ABuStore import get_and_store_jijin_data
from abupy.MarketBu import ABuSymbolPd
from abupy.MarketBu.ABuDataCache import load_kline_df
from abupy.CoreBu.ABuEnv import EDataCacheType, EMarketDataFetchMode, EMarketSourceType, EMarketTargetType
from pandas.plotting import register_matplotlib_converters

from abupy.UtilBu import ABuRegUtil
ABuEnv.g_data_fetch_mode = EMarketDataFetchMode.E_DATA_FETCH_FORCE_NET  #强制从网络获取
# ABuEnv.g_data_fetch_mode = EMarketDataFetchMode.E_DATA_FETCH_FORCE_LOCAL  #强制本地，可多线程

ABuEnv.g_market_source = EMarketSourceType.E_MARKET_SOURCE_tx  #作用同上点击效果。腾讯数据源(美股，A股，港股)
ABuEnv.g_market_target = EMarketTargetType.E_MARKET_TARGET_CN
register_matplotlib_converters()

stock_code = '600988' #中欧医疗健康混合A
# stock_code = '162703' #广发小盘成长混合（LOF）
stock_name = 'xxx'
start_date = '2020-02-12'  # 设置起始日期
end_date = '2020-02-20'  # 设置终止日期
# stock_k = ts.get_hist_data(stock_code, start=start_date, end=end_date)

from abupy import code_to_symbol, IndexSymbol, EMarketDataSplitMode, EMarketDataFetchMode, EMarketSourceType

sh_stock_code = code_to_symbol(stock_code).value
stock_k, df_req_start, df_req_end = load_kline_df(sh_stock_code) #读取已存在的数据
kl_pd = ABuSymbolPd.make_kl_df(stock_code, data_mode=EMarketDataSplitMode.E_DATA_SPLIT_UNDO,
                                               benchmark=None, n_folds=1)
# temp1 = ts.get_hist_data('161005',start='2020-02-03',end='2020-02-22',ktype='D')
#ktype：数据类型，D=日k线 W=周 M=月 5=5分钟 15=15分钟 30=30分钟 60=60分钟，默认为D

kl_pd = get_and_store_jijin_data('519005')
kl_pd = kl_pd.tail(300)
kl_pd['date2'] = [int(datetime.datetime.strptime(dateTemp, '%Y-%m-%d').strftime('%Y%m%d')) for dateTemp in kl_pd.date]
print('test')

# a = datetime.datetime.strptime('2018-1-8', '%Y-%m-%d')

# temp = kl_pd.date2.values
plt.plot(kl_pd.date, kl_pd.dwjz)
plt.show()

# benchmark_xd = benchmark_df[start_key:end_key + 1]
benchmark = IndexSymbol.SH #获取大盘数据。
bench_pd = ABuSymbolPd.make_kl_df(benchmark, data_mode=EMarketDataSplitMode.E_DATA_SPLIT_SE,
                                        n_folds=None,
                                        start='2019-04-20', end='2019-05-01')
ang = ABuRegUtil.calc_regress_deg(bench_pd.close, show=True) * 100

ABuMarketDrawing.plot_candle_stick(kl_pd.index, kl_pd['open'].values, kl_pd['high'].values, kl_pd['low'].values,
                      kl_pd['close'].values, kl_pd['volume'].values, None,
                      'title', day_sum=False, html_bk=False, save=False,kl_pd=kl_pd, minute=False,bench=kl_pd)

# date, p_open, high, low, close, volume, view_index,
#                                    symbol, day_sum, html_bk, save, minute=False,kl_pd=None)


print('test')
def get_EMA(cps, days):
    emas = cps.copy()  # 创造一个和cps一样大小的集合
    for i in range(len(cps)):
        if i == 0:
            emas[i] = cps[i]
        if i > 0:
            emas[i] = ((days - 1) * emas[i - 1] + 2 * cps[i]) / (days + 1)
    return emas



stock_k['ema10'] = get_EMA(stock_k.close,10)
stock_k['ema60'] = get_EMA(stock_k.close,60)
plt.plot(stock_k.ema10,c='r')
plt.plot(stock_k.ema60,c='green')
plt.plot(stock_k.close,c='blue')
plt.show()

from abupy import ABuMarketDrawing
ABuMarketDrawing.plot_candle_form_klpd(stock_k,
                                       html_bk=False)

print('test2')


stock_table = pd.DataFrame()

df = ts.get_tick_data(stock_code, date='2019-2-25', src='tt')  #获取详细数据

for current_date in stock_k.index:
    # 通过loc选中K线图中对应current_date这天的数据
    current_k_line = stock_k.loc[current_date]

    # 提取这一天前10分钟股票信息
    df = ts.get_tick_data(stock_code, date=current_date, src='tt')
    df['time'] = pd.to_datetime(current_date + ' ' + df['time'])
    t = pd.to_datetime(current_date).replace(hour=9, minute=40)
    df_10 = df[df.time <= t]
    vol = df_10.volume.sum()  # 通过sum()函数求和

    # 将数据信息放入字典中
    current_stock_info = {
        '名称': stock_name,
        '日期': pd.to_datetime(current_date),
        '开盘价': current_k_line.open,
        '收盘价': current_k_line.close,
        '股价涨跌幅(%)': current_k_line.p_change,
        '10分钟成交量': vol
    }
    # 通过append的方式增加新的一行，忽略索引
    stock_table = stock_table.append(current_stock_info, ignore_index=True)
stock_table = stock_table.set_index('日期')

# 设置列的顺序
order = ['名称', '开盘价', '收盘价', '股价涨跌幅(%)', '10分钟成交量']
stock_table = stock_table[order]

'''2.下面开始获得股票衍生变量数据'''

# 通过公式1获取成交量涨跌幅
stock_table['昨日10分钟成交量'] = stock_table['10分钟成交量'].shift(-1)
stock_table['成交量涨跌幅1(%)'] = (stock_table['10分钟成交量']-stock_table['昨日10分钟成交量'])/stock_table['昨日10分钟成交量']*100

# 通过公式2获得成交量涨跌幅
ten_mean = stock_table['10分钟成交量'].sort_index().rolling(10, min_periods=1).mean()
stock_table['10分钟成交量10日均值'] = ten_mean
stock_table['成交量涨跌幅2(%)'] = (stock_table['10分钟成交量']-stock_table['10分钟成交量10日均值'])/stock_table['10分钟成交量10日均值']*100

print(stock_table)


'''3.通过相关性分析选取合适的衍生变量'''
from scipy.stats import pearsonr
# 通过公式1计算的相关性
corr = pearsonr(abs(stock_table['股价涨跌幅(%)'][:-1]), abs(stock_table['成交量涨跌幅1(%)'][:-1]))
print('通过公式1计算的相关系数r值为' + str(corr[0]) + '，显著性水平P值为' + str(corr[1]))

# 通过公式2计算的相关性
corr = pearsonr(abs(stock_table['股价涨跌幅(%)']), abs(stock_table['成交量涨跌幅2(%)']))
print('通过公式2相关系数r值为' + str(corr[0]) + '，显著性水平P值为' + str(corr[1]))


axs=None
drawer = plt if axs is None else axs
drawer.plot(stock_table['股价涨跌幅(%)']*100,c='r')
drawer.plot(stock_table['成交量涨跌幅2(%)'],c='g')
plt.xlabel('time')
plt.ylabel('close')
plt.title('wan ke A')
plt.grid(True)
plt.show()


