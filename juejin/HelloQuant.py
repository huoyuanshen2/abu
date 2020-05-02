# coding=utf-8
from __future__ import print_function, absolute_import, unicode_literals
from gm.api import *
def init(context):
    schedule(schedule_func=algo, date_rule='1d', time_rule='14:50:00')
def algo(context):
    # 1、获取基金数据，获取大盘数据 老方法
    # 2、数据处理，并还原图像。
    # 3、找到基本规律，构建买入信号。
    # 4、构建卖出信号
    # 5、查看回测结果
    # 获取基金基础数据
    data = get_instrumentinfos(symbols=None, exchanges=None, sec_types=SEC_TYPE_FUND, names=None, fields=None, df=True)
    data = data[data.sec_name.str.contains('LOF')]
    symbol = data[data.sec_id == '501000'].symbol.values[0]
    kl_pd = history(symbol, "1d", "2019-01-01", "2020-01-01", fields=None, skip_suspended=True,
            fill_missing=None, adjust=ADJUST_PREV, adjust_end_time='', df=True)

    # 获取大盘数据
    # data = get_instruments(symbols=data.symbol.value[0], exchanges=None, sec_types=2, names=None, skip_suspended=True, skip_st=True,
    #                        fields=None, df=True)
    data = get_history_instruments(symbols=data.symbol.values[0], fields=None, start_date='2020-01-01', end_date='2020-01-29', df=True)

    # data = history_n(symbol="SZSE.003095", frequency="1d", count=100, end_time="2020-02-20", fields=None,
    #                  fill_missing="last", adjust=ADJUST_PREV, df=True)
    #
    # # 购买200股浦发银行股票
    order_volume(symbol='SHSE.501000', volume=200, side=OrderSide_Buy, order_type=OrderType_Market, position_effect=PositionEffect_Open, price=0)

if __name__ == '__main__':
    run(strategy_id='9de5f1cb-5a90-11ea-b515-02004c4f4f50', filename='HelloQuant.py', mode=MODE_BACKTEST, token='8e1026d2dfd455be2e1f239e50004b35a481061e',
        backtest_start_time='2016-06-17 13:00:00', backtest_end_time='2017-08-21 15:00:00')
