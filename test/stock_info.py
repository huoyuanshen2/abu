# -*- encoding:utf-8 -*-
from __future__ import print_function
from __future__ import division

from abupy.MarketBu.ABuSymbolStock import query_stock_info
# 不要在abupy之外再建立包结构
# noinspection PyUnresolvedReferences
# import widget_base
from abupy.UtilBu.ABuStrUtil import to_unicode
from abupy import WidgetSearchStockInfo, WidgetStockInfo

widget = WidgetSearchStockInfo()
# widget.search_input="双汇"
stock_info = query_stock_info("000895")
# co_name_str =公司名称
co_name_str = WidgetStockInfo._combine_stock_name(WidgetStockInfo,stock_info)
# description=u'公司简介:'
description = to_unicode(stock_info.co_intro.values[0])
pv_dict = {
            'pe_s_d': u"市盈率(静)/(动):",
            'pb_d': u"市净率(动):",
            'pb_MRQ': u"市净率MRQ:",
            'ps_d': u"市销率(动):",
            'ps': u"市销率:",
            'pe_s': u"市盈率(静):"}
print(u'公司名称:'+co_name_str)
print(u'公司简介:'+to_unicode(stock_info.co_intro.values[0]))
print(u"市盈率(静)/(动):"+stock_info.pe_s_d.values[0])
print(u"市净率(动):"+stock_info.pb_d.values[0])
# print(u"市净率MRQ:"+stock_info['pb_d'].values[0])
print(u"市销率(动):"+stock_info.ps_d.values[0])

# print(u"市销率:"+stock_info.ps.values[0])
# st1_temp = stock_info['ps']
# if len(st1_temp):
#     print(u"市销率:" + stock_info.ps.values[0])
print(u"市盈率(静):"+stock_info.pe_s.values[0])






description=3


