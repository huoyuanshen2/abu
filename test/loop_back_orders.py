import time


from abupy import AbuFactorBuyBreak, AbuFactorAtrNStop, AbuFactorPreAtrNStop, AbuFactorCloseAtrNStop, AbuFactorSellNDay, \
    AbuDownUpTrend, ABuGridHelper, WrsmScorer, GridSearch, abu, AbuMetricsBase, AbuSlippageBuyMean, EStoreAbu, \
    code_to_symbol, AbuBenchmark
from abupy.BetaBu import ABuPositionBase
from abupy.BetaBu.ABuRatePosition import AbuRatePosition
from abupy.CoreBu import ABuEnv
from abupy.CoreBu.ABuEnv import EDataCacheType, EMarketDataFetchMode, EMarketSourceType, EMarketTargetType
import numpy as np
import scipy.stats as scs
import tushare as ts
import datetime
import pandas as pd
import matplotlib.pyplot as plt
# 基于结果orders对象进行二次处理。
#----------------------------资金、股池、基础因子-----------------------------
from abupy.CoreBu.ABuStore import store_abu_result_out_put, store_abu_result_tuple, load_abu_result_tuple, \
    store_python_obj, load_python_obj, get_and_store_stock_detail_data
from abupy.MarketBu.ABuDataCache import load_kline_df
from abupy.TradeBu.ABuTradeDrawer import plot_result_price, plot_result_volumn, plot_result_volumn_sum

ABuEnv.date_str = "_2020_03_27"

if __name__ == '__main__':    #多线程必须内容，不可删除。
    score_tuple_array = load_python_obj('score_tuple_array'+ABuEnv.date_str)
#-------权重方式开始------
    if score_tuple_array != None:
        scorer = WrsmScorer(score_tuple_array, weights=[1, 0, 0, 0])
        test = scorer.score_pd
        scorer_returns_max = scorer.fit_score()
        best_result_tuple = score_tuple_array[scorer_returns_max.index[-1]]
        orders_pd = best_result_tuple.orders_pd[0:4000]
        key_value = 1000 * 10000
        order_pds_win = None
        order_pds_loss = None
        order_pds = None
        for key,order in orders_pd.iterrows() :
            # break
            buy_date = order.buy_date
            stock_code = order.symbol #股票代码
            sh_stock_code = code_to_symbol(stock_code).value
            kl_pd, df_req_start, df_req_end = load_kline_df(sh_stock_code)

            kl_pd_key = kl_pd[kl_pd.date == buy_date]['key'].values[0]
            kl_pd_yestaday = kl_pd.iloc[kl_pd_key-1]  # 获得买入前一天数据
            yestadayDate = kl_pd_yestaday.date

            timeArray = time.strptime(str(int(yestadayDate)), "%Y%m%d")
            date_str = time.strftime("%Y-%m-%d", timeArray)

            stock_detail_data = get_and_store_stock_detail_data(stock_code, date_str)
            if stock_detail_data is None:
                 continue
            today_max_data = stock_detail_data[stock_detail_data.price == kl_pd_yestaday.close]
            con_all = stock_detail_data.shape[0]
            con_max = today_max_data.shape[0]
            maxRate = (con_max * 100) / con_all
            order['maxRate'] = maxRate
            order_pd = pd.DataFrame(order).T
            if  maxRate <= 2 :
                order_pds = order_pd.append(order_pds)
                if order.sell_type == 'win':
                    order_pds_win = order_pd.append(order_pds_win)
                elif order.sell_type == 'loss':
                    order_pds_loss = order_pd.append(order_pds_loss)
                pass

        # plt.hist(order_pds_win.maxRate,bins=50,normed = True,alpha=0.5,color='red')
        # plt.hist(order_pds_loss.maxRate, bins=50, normed=True, alpha=0.5, color='green')
        # plt.show()
        print(len(order_pds))
        print(len(order_pds_win))
        print(len(order_pds_loss))

        AbuMetricsBase.show_orders(order_pds, best_result_tuple.action_pd,
                                    best_result_tuple.capital, best_result_tuple.benchmark, only_info=True)
    else:
        print("没有订单生成！")
