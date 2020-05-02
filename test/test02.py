import tushare as ts
import pandas as pd
import matplotlib.pyplot as plt
from pandas import merge

from abupy import EStoreAbu
from abupy.CoreBu.ABu import load_abu_result_tuple

import matplotlib.pyplot as plt
import numpy as np

# x = [8450.0, 8061.0, 7524.0, 7180.0, 8247.0, 8929.0, 8896.0, 9736.0, 9658.0, 9592.0]
# x = [8450.0, 8061.0, 7524.0, 7180.0, 8247.0, 8929.0, 8896.0, 9736.0, 9658.0, 9592.0]
y = [-1.416,-0.597, 0.208, 0.612, 0.913, 1.221, 1.311, 1.614,-0.697,-1.231]
x = range(len(y))
# y = range(len(x))

best_fit_line = np.poly1d(np.polyfit(y, x, 1))(y)

slope = (y[-1] - y[0]) / (x[-1] - x[0])
angle = np.arctan(slope)

print('slope: ' + str(slope))
print ('angle: ' + str(angle))
angle2 = np.rad2deg(np.arctan2(y[-1] - y[0], x[-1] - x[0]))
print('angle2:{}'.format(angle2))
plt.figure(figsize=(8,6))
# plt.plot(x)
plt.plot(x,best_fit_line, '--', color='r')
plt.show()


result_touple = load_abu_result_tuple(n_folds=None, store_type=EStoreAbu.E_STORE_NORMAL, custom_name=None)
print(result_touple.orders_pd.head(5))
orders_pd = result_touple.orders_pd
industry_data = ts.get_industry_classified()

orders_pd = merge(orders_pd, industry_data, how='left', on=None, left_on='symbol', right_on='code',
      left_index=False, right_index=False, sort=True,
       copy=True, indicator=False)

# group_orders = orders_pd.groupby(['c_name','result']).mean()
# counts = orders_pd.buy_price.groupby(['c_name','result']).value_counts().to_frame('count').reset_index()
counts = orders_pd.groupby(by=['c_name','result'], as_index=False).count()
# 行业有正负，正数多
#没有负数
win_orders_pd = orders_pd[orders_pd.result == 1]
loss_orders_pd = orders_pd[orders_pd.result == -1]

max_buy_price = orders_pd.buy_price.max()
min_buy_price = orders_pd.buy_price.min()

# orders_pd.buy_price.plot(kind='hist', xlim=(min_buy_price,max_buy_price), bins=50)
win_orders_pd.buy_price.plot(kind='hist', xlim=(min_buy_price,max_buy_price), bins=50)
plt.show()
loss_orders_pd.buy_price.plot(kind='hist', xlim=(min_buy_price,max_buy_price), bins=50)
plt.xlabel('buy_price')
plt.ylabel('count')
plt.title('wan ke A')
plt.show()
win_orders_pd.buy_price.dropna().plot(kind='kde', xlim=(min_buy_price,max_buy_price))
loss_orders_pd.buy_price.dropna().plot(kind='kde', xlim=(min_buy_price,max_buy_price))

plt.show()