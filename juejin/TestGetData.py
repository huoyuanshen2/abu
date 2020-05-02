from gm.api import *
import talib
import numpy as np
set_token("8e1026d2dfd455be2e1f239e50004b35a481061e")

# data=history_n(symbol="SZSE.003095",frequency="1d",count=100,end_time="2017-12-31",fields="close",fill_missing="last",adjust=ADJUST_PREV,df=True)
# 获取基金大概信息
data = get_instrumentinfos(symbols='SHSE.003095', exchanges=['SHSE','SZSE'], sec_types=2, names=None, fields=None, df=True)
#查询标的基本信息
get_instrumentinfos(symbols=None, exchanges=None, sec_types=None, names=None, fields=None, df=False)
# 查询标的最新信息
get_instruments(symbols=None, exchanges=None, sec_types=None, names=None, skip_suspended=True, skip_st=True, fields=None, df=False)

#查询历史数据，非业务数据
get_history_instruments(symbols, fields=None, start_date=None, end_date=None, df=False)
#查询历史业务数据
history(symbol, frequency, start_time, end_time, fields=None, skip_suspended=True,
        fill_missing=None, adjust=ADJUST_PREV, adjust_end_time='', df=False)
#查询目标日最近N天数据
history_n(symbol, frequency, count, end_time=None, fields=None, skip_suspended=True,
          fill_missing=None, adjust=ADJUST_PREV, adjust_end_time='', df=False)

# data = get_instrumentinfos(symbols=['SHSE.000001'],df=True)
print(data)
print("test")