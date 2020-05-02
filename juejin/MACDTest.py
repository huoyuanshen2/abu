# coding=utf-8
from __future__ import print_function, absolute_import, unicode_literals
import numpy as np
import talib
from gm.api import *
def init(context):
    context.symbol="SZSE.300296"
    context.frequency="1d"
    context.fields = "open,high,low,close"
    context.volume = 200
    schedule(schedule_func=algo, date_rule='1d', time_rule='09:35:00')
def algo(context):
    now = context.now
    last_day = get_previous_trading_date("SZSE",now)
    data = history_n(symbol=context.symbol,frequency=context.frequency,count=35,end_time=last_day
                     ,fields=None,fill_missing="last",adjust=ADJUST_PREV,df=True)
    open = np.asarray((data["open"].values))
    high = np.asarray((data["high"].values))
    low = np.asarray((data["low"].values))
    close = np.asarray((data["close"].values))
    macd,_,_ = talib.MACD(close) #此处有问题，未找到源码
    macd = macd[-1]


    if macd > 0:
        order_volume(symbol=context.symbol,volume=context.volume,side=PositionSide_Long,order_type=OrderType_Market,position_effect=PositionEffect_Open)
        print("买入")
    elif macd < 0:
        print("卖出")
        order_volume(symbol=context.symbol, volume=context.volume, side=PositionSide_Short, order_type=OrderType_Market,
                     position_effect=PositionEffect_Close)

if __name__ == '__main__':
    run(strategy_id='9de5f1cb-5a90-11ea-b515-02004c4f4f50', filename='MACDTest.py', mode=MODE_BACKTEST,
        token='8e1026d2dfd455be2e1f239e50004b35a481061e',
        backtest_start_time='2017-01-01 09:00:00', backtest_end_time='2017-12-31 15:00:00',
        backtest_initial_cash=20000,backtest_adjust=ADJUST_PREV)
