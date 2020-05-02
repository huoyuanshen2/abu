from __future__ import print_function, absolute_import, unicode_literals

import datetime

from gm.api import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from abupy.UtilBu import ABuDateUtil

set_token("8e1026d2dfd455be2e1f239e50004b35a481061e")
data = get_instrumentinfos(symbols=None, exchanges=None, sec_types=1, names=None, fields=None, df=True)
# symbol = data[data.sec_id == '002719'].symbol.values[0] #麦趣尔
# symbol = data[data.sec_id == '002910'].symbol.values[0] #庄园牧场
symbol = data[data.sec_id == '600021'].symbol.values[0] #庄园牧场

# ------------------正太数据绘图开始
import tushare as ts
import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

# pro = ts.pro_api('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
start_date = "2018-03-18" #包含
end_date = "2020-03-19" #不包含

# df = pd.a[1,2,3,4,5,6,7,8,9,10]
df1 = [1,2,3,4,5,6,7,8,9,10]

df=pd.DataFrame(data=df1,columns=['num'])
print(df.head())
values = df['num']
shape, loc, scale = scipy.stats.lognorm.fit(values)
x = np.linspace(values.min(), values.max(), len(values))
pdf = scipy.stats.lognorm.pdf(x, shape, loc=loc, scale=scale)
label = 'mean=%.4f, std=%.4f, shape=%.4f' % (loc, scale, shape)

values.hist(values, bins=30, density=True)
values.plot(x, pdf, 'r-', lw=2, label=label)
values.legend(loc='best')
plt.show()
df_tw = df
df_tw.hist(bins=5, alpha=0.9)
df_tw.plot(kind='kde', secondary_y=True)
plt.show()
print(df.head())
pass


# ticker_data = history(symbol, "1d", start_date, end_date, fields=None, skip_suspended=True,
#         fill_missing=None, adjust=ADJUST_PREV, adjust_end_time='', df=True)
# print('数据量：',len(ticker_data))
# print(ticker_data.head(5))
# ticker_data['trade_date'] = pd.to_datetime(ticker_data['trade_date'],format='%Y%m%d')
# ticker_data.set_index('trade_date', inplace=True)
# returns = ticker_data["close"].pct_change().dropna()

# plt.figure(figsize=(15, 5))
# plt.title("股票代码:600377 - 宁沪高速", weight='bold')
# ticker_data['close'].plot()
# plt.show()
# ticker_data.set_index('bob', inplace=True)
# ticker_data['close'].plot()
# plt.show()
#
# _,pvalue= scipy.stats.jarque_bera(returns)
# print(pvalue)
# if pvalue > 0.05:
#     print ('数据服从正态分布')
# else:
#     print ('数据不服从正态分布')
#
# fig, ax = plt.subplots(figsize=(10, 6))
#
# values = ticker_data["close"]
#
# shape, loc, scale = scipy.stats.lognorm.fit(values)
# x = np.linspace(values.min(), values.max(), len(values))
# pdf = scipy.stats.lognorm.pdf(x, shape, loc=loc, scale=scale)
# label = 'mean=%.4f, std=%.4f, shape=%.4f' % (loc, scale, shape)
#
# ax.hist(values, bins=30, density=True)
# ax.plot(x, pdf, 'r-', lw=2, label=label)
# ax.legend(loc='best')
# plt.show()
#
# values = returns
# x = np.linspace(values.min(), values.max(), len(values))
#
# loc, scale = scipy.stats.norm.fit(values)
# param_density = scipy.stats.norm.pdf(x, loc=loc, scale=scale)
# label = '均值=%.4f, 标准差=%.4f' % (loc, scale)
#
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.hist(values, bins=30, density=True)
# ax.plot(x, param_density, 'r-', label=label)
# ax.legend(loc='best')
# plt.show()

# ---------------分时数据绘图开始--------------------
start_date = "2020-03-18" #包含
end_date = "2020-03-19" #不包含

kl_pd = history(symbol, "60s", start_date, end_date, fields=None, skip_suspended=True,
        fill_missing=None, adjust=ADJUST_PREV, adjust_end_time='', df=True)

bench_kl_pd = history('SHSE.000001', "60s", start_date, end_date, fields=None, skip_suspended=True,
        fill_missing=None, adjust=ADJUST_PREV, adjust_end_time='', df=True)
print("len(bench_kl_pd) - len(kl_pd) = {}".format((len(bench_kl_pd) - len(kl_pd))))
if (len(bench_kl_pd) - len(kl_pd)) < 0:
    print("(len(bench_kl_pd) - len(kl_pd)) < 0 ,exit.")
    exit

bench_kl_pd = bench_kl_pd[bench_kl_pd['bob'].isin(kl_pd['bob'].tolist())]
bench_kl_pd.index = np.arange(0,len(bench_kl_pd))

kl_pd['date'] = kl_pd['bob'].apply(lambda x: ABuDateUtil.date_time_str_to_int(str(x)))
kl_pd['time'] = kl_pd['bob'].apply(lambda x: ABuDateUtil.date_time_str_to_time_str(str(x)))

kl_pd_time = kl_pd[kl_pd.time == '093000']

kl_pd['p_change'] = (kl_pd.close - kl_pd['close'][0])/kl_pd['close'][0]
bench_kl_pd['p_change'] = (bench_kl_pd.close - bench_kl_pd['close'][0])/bench_kl_pd['close'][0]
kl_pd['p_change_update'] = (kl_pd.p_change - bench_kl_pd.p_change)

window_volume = 30
window_close = 30
kl_pd['p_change_5ma'] = kl_pd.p_change.rolling(window = window_close).mean()
kl_pd['p_change_update_5ma'] = kl_pd.p_change_update.rolling(window = window_close).mean()
bench_kl_pd['p_change_5ma'] = bench_kl_pd.p_change.rolling(window = window_close).mean()

kl_pd['volume_ma'] = kl_pd.volume.rolling(window = window_volume).mean()


kl_pd['p_change_5ma_up_rate'] = (kl_pd.p_change_5ma -kl_pd.p_change_5ma.shift(5))
kl_pd['p_change_5ma_up_rate_chu_time'] = (kl_pd.p_change_5ma -kl_pd.p_change_5ma.shift(5))/5

kl_pd['p_change_5ma_up_rate_chu_time_2'] = 5/(kl_pd.p_change_5ma -kl_pd.p_change_5ma.shift(5))

kl_pd['p_change_update_5ma_up_rate'] = (kl_pd.p_change_update_5ma -kl_pd.p_change_update_5ma.shift(5))
bench_kl_pd['p_change_5ma_up_rate'] = (bench_kl_pd.p_change_5ma - bench_kl_pd.p_change_5ma.shift(5))
kl_pd['zero_line'] = 0

kl_pd['volume_ma_up_rate'] = (kl_pd.volume_ma -kl_pd.volume_ma.shift(5))
max_p_change = kl_pd['p_change_5ma_up_rate'].max()
max_volume = kl_pd['volume_ma_up_rate'].max()

vs_rate = max_p_change/max_volume
kl_pd['volume_ma_up_rate'] = (kl_pd.volume_ma -kl_pd.volume_ma.shift(5)) * vs_rate


# plt.plot(kl_pd.index, kl_pd['p_change'], label='p_change', color='blue') #基础p_change
# plt.plot(kl_pd.index, bench_kl_pd['p_change'], label='bench_p_change', color='green') #大盘p_change
# plt.plot(kl_pd.index, kl_pd['p_change_5ma'], label='close60', color='red') #基础p_change均线
# plt.plot(kl_pd.index, bench_kl_pd['p_change_5ma'], label='close60', color='red') #基础大盘p_change均线
# plt.plot(kl_pd.index, kl_pd['p_change_update'],'--', label='p_change_update', color='red') #修正后涨跌幅

plt.plot(kl_pd.index, kl_pd['p_change_5ma_up_rate'], label='close60', color='red') #基础均线增长斜率
plt.plot(kl_pd.index, kl_pd['p_change_5ma_up_rate_chu_time'], label='chu_time', color='blue') #基础均线增长斜率
# plt.plot(kl_pd.index, kl_pd['p_change_update_5ma_up_rate'], '--',label='close60', color='blue') #修正均线增长斜率
# plt.plot(bench_kl_pd.index, bench_kl_pd['p_change_5ma_up_rate'], label='close60', color='green') #大盘增长斜率


plt.plot(kl_pd.index, kl_pd['zero_line'], label='0_line', color='black') #0线
# plt.vlines(kl_pd_time.index, middle_line, middle_line+0.2,color="black") #日期分割线
# plt.plot(kl_pd.index, kl_pd['volume_ma'], label='volume_ma', color='blue') #量均值
# plt.plot(kl_pd.index, kl_pd['volume_ma_up_rate'], '--',label='up_rate', color='red') #量增长斜率
plt.title(str(symbol) + '_'+start_date)
plt.legend(loc='upper left')
plt.show()

print(kl_pd)
