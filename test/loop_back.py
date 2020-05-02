import os
import pandas as pd
import abupy
from abupy import WidgetRunLoopBack, AbuDoubleMaBuy, ABuFileUtil, AbuPickStockPriceMinMax
from abupy.BetaBu import ABuPositionBase
from abupy.BetaBu.ABuRatePosition import AbuRatePosition
from abupy.CoreBu import ABuEnv
from abupy.CoreBu.ABuEnv import EDataCacheType, EMarketDataFetchMode, EMarketSourceType ,EMarketTargetType
import tushare as ts

from abupy.FactorBuyBu.ABuFactorBuyBigWave2 import AbuBigWave2
from abupy.FactorBuyBu.ABuFactorBuyIndustry import AbuIndustryRateBuy

widget = WidgetRunLoopBack()
ABuEnv.g_cpu_cnt = 1 #并发运行线程数
ABuEnv.draw_order_num = 5 #要绘制的订单数

#----------------------------基础设置-----------------------------
ABuPositionBase.g_default_pos_class = {'class':AbuRatePosition,'base_rate':0.5}



widget.tt.cash.value = 100000 #初始资金
widget.tt.time_mode.value = 0 #时间模式0：使用回测年数 1：使用开始结束日期
widget.tt.run_years.value = 1 #回测年数 1-6年
widget.tt.start.value = '2019-07-26' #开始日期
widget.tt.end.value = '2019-12-26' #结束日期
widget.tt.date_mode.value = u'开放数据模式' #数据模式。 或者： u'开放数据模式'
ABuEnv.disable_example_env_ipython(show_log=False) #网络模式本地网络结合

widget.tt.store_mode = EDataCacheType.E_DATA_CACHE_CSV.value  #缓存模式 csv模式(推荐)
ABuEnv.g_data_fetch_mode = EMarketDataFetchMode.E_DATA_FETCH_NORMAL  #本地结合网络获取
# ABuEnv.g_data_fetch_mode = EMarketDataFetchMode.E_DATA_FETCH_FORCE_NET  #强制从网络获取
widget.tt.date_source = EMarketSourceType.E_MARKET_SOURCE_tx.value #腾讯数据源(美股，A股，港股)
ABuEnv.g_market_source = EMarketSourceType.E_MARKET_SOURCE_tx  #作用同上点击效果。百度数据源(美股，A股，港股)
#----------------------------股池设置-----------------------------
# widget.sc.choice_symbols.options = list(set(['000002','000895'] + list(widget.sc.choice_symbols.options)))
industry_data = ts.get_industry_classified() #获取行业信息
industry_data = industry_data[~industry_data.name.str.contains('ST')]
industry_data2 = industry_data['code'][industry_data.c_name == '公路桥梁'].head(10) #过滤
# industry_data2 = industry_data['code'][industry_data['c_name'].isin(['公路桥梁','综合行业','家电行业'])]

widget.sc.choice_symbols.options = list(set(industry_data2))
widget.sc.market = EMarketTargetType.E_MARKET_TARGET_CN.value #大盘市场选项 A股 。大盘市场设置只影响收益对比标尺
ABuEnv.g_market_target = EMarketTargetType.E_MARKET_TARGET_CN  #上面按钮触发赋值。 大盘市场选项 A股 。

#----------------------------买入因子-----------------------------
# widget.bf
#双均线买
# factor_dict1 = {'class': AbuDoubleMaBuy, 'slow': -1, 'fast': -1} #策略对象，点击添加策略按钮后触发
# widget.bf.factor_dict["动态双均慢动态快动态买入"] = factor_dict1 #策略对象，点击添加策略按钮后触发

#长跌短涨
xd_value = 5 #短线周期：比如20，30，40天,短线以及突破参数
past_factor_value = 4 #长线乘数：短线基础 x 长线乘数 = 长线周期
down_deg_threshold_value = -3 #拟合趋势角度阀值：如-2,-3,-4
factor_dict2 = {'class': abupy.AbuDownUpTrend, 'xd': xd_value,
                       'past_factor': past_factor_value, 'down_deg_threshold': down_deg_threshold_value}
factor_desc_key = u'长线{}下跌短线{}上涨角度{}'.format(xd_value * past_factor_value, xd_value, down_deg_threshold_value)
# widget.bf.factor_dict[factor_desc_key] = factor_dict2
# widget.bf.selected_factors.options = list(widget.bf.factor_dict.keys()) #仅展示效果。

#行业比率
# factor_dict_AbuIndustryRateBuy = {'class': AbuIndustryRateBuy, 'changeRate': 1, 'targetCountRate': 20}
# widget.bf.factor_dict["行业比率买入因子"] = factor_dict_AbuIndustryRateBuy #策略对象，点击添加策略按钮后触发

#N天趋势突破参照大盘
# factor_dict_AbuIndustryRateBuy = {'class': AbuIndustryRateBuy, 'changeRate': 1, 'targetCountRate': 20}
# widget.bf.factor_dict["行业比率买入因子"] = factor_dict_AbuIndustryRateBuy
# xd：突破周期
# polt:大盘走势拟合次数阀值，poly大于阀值＝震荡
# factor_dict_AbuIndustryRateBuy = {'class': abupy.AbuSDBreak, 'xd': 21, 'poly': 2}
factor_dict_AbuIndustryRateBuy = {'class': AbuBigWave2, 'xd': 21,'bigWave2Able':1,'bigWave':0.1}
widget.bf.factor_dict["xd拟合N天趋势突破参照大盘"] = factor_dict_AbuIndustryRateBuy

#----------------------------选股因子-----------------------------
factor_dict = {'class': AbuPickStockPriceMinMax, #选股策略
                       'xd': 252, #周期1-252，默认252
                       'reversed': False, #是否翻转
                       'threshold_price_min': 15, #使用最小阀值  '设定选股价格最小阀值，默认15'
                       'threshold_price_max': 50} #使用最大阀值 设定选股价格最大阀值，默认50
factor_desc_key = u'价格选股最大:{}最小:{},周期:{},反转:{}'.format(50, 15, 252, False)
# for pickStock in widget.ps.ps_array:
#     if(isinstance(pickStock, abupy.WidgetBu.ABuWGPickStock.PSPriceWidget)): #根据类型，进行参数传递。同点击添加策略按钮
#         pickStock.wg_manager.add_factor(factor_dict, factor_desc_key)
# self.ps_array = [] #包含4个对象。
#----------------------------卖出因子-----------------------------
# N天卖出
sell_n = 50 #设定买入后只持有天数，默认1
is_sell_today = True #设定买入n天后，当天还是隔天卖出
factor_dict_sell = {'class': abupy.AbuFactorSellNDay, 'sell_n': sell_n,
                       'is_sell_today': is_sell_today}
factor_desc_sell_key = u'持有{}天{}卖出'.format(sell_n, u'当天' if is_sell_today else u'隔天')
widget.sf.factor_dict[factor_desc_sell_key] = factor_dict_sell

#----------------------------开始回测-----------------------------
if __name__ == '__main__':    #多线程必须内容，不可删除。
  widget.run_loop_back(2)







