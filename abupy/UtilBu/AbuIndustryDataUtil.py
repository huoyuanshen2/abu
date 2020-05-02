# -*- encoding:utf-8 -*-
"""
    abupy中行业数据辅助工具

    eg:
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import datetime
from collections import Iterable

import logging
import tushare as ts
import numpy as np
import pandas as pd

# from abupy import AbuBenchmark, AbuCapital, AbuKLManager
from abupy.CoreBu.ABuStore import store_csv_data, load_csv_data, store_python_obj, load_python_obj
from abupy.SimilarBu.ABuCorrcoef import ECoreCorrType, corr_matrix
import abupy
from ..CoreBu import ABuEnv
from ..CoreBu.ABuPdHelper import pd_resample
from ..TradeBu import AbuBenchmark,AbuCapital,AbuKLManager
__author__ = '阿布'
__weixin__ = 'abu_quant'

log_func = logging.info if ABuEnv.g_is_ipython else print


def getIndustryData(kl_pd_manager,concept=None):
    '''行业(或概念)数据规整
                000895  000666  ...
    20190101    0.1     -0.2    ...
    20190102    6       8       ...
    ...
    '''
    # 根据天数进行判断，是否当前有可用数据。有则复用，否则新建。
    # date_dir = datetime.datetime.now().strftime("%Y_%m_%d")
    dataDicFileName = 'industryPchangeAllDataDic_'+concept
    industryPchangeAllDataDic = load_python_obj(dataDicFileName)
    # industryDataFileName = date_dir+ '_industry_data'
    industryDataFileName = 'industry_data'
    industry_data = load_csv_data(industryDataFileName)
    industryPearsonrDataDicFileName = 'industryPearsonrDataDic'
    industryPearsonrDataDic = load_python_obj(industryPearsonrDataDicFileName)

    if (industryPchangeAllDataDic is not None) and (industry_data is not None) and (industryPearsonrDataDic is not None):
        ABuEnv.industryPchangeAllDataDic = industryPchangeAllDataDic
        ABuEnv.industry_data = industry_data
        ABuEnv.industryPearsonrDataDic = industryPearsonrDataDic
        return

    industryPchangeAllDataDic = dict()
    industry_data = ts.get_industry_classified()
    industry_data = industry_data[~industry_data.name.str.contains('ST')]
    kl_pd_dick = kl_pd_manager.pick_kl_pd_dict['pick_time']
    for stockCode, stockData in kl_pd_dick.items():
        if stockData is None:
            continue
        stockIndustryNames =  pd.Series([concept]) if concept is not None else industry_data['c_name'][industry_data.code == stockCode]
        for _, stockIndustryName in stockIndustryNames.items():
            if stockIndustryName in industryPchangeAllDataDic.keys():
                left = industryPchangeAllDataDic[stockIndustryName]
            else:
                left = pd.DataFrame({'date': stockData.date})
            right = pd.DataFrame({stockCode: stockData.p_change})
            industryPchangeAllDataDic[stockIndustryName] = pd.merge(left, right, left_index=True, right_index=True,
                                                                        how='outer')
    ABuEnv.industryPchangeAllDataDic = industryPchangeAllDataDic
    ABuEnv.industry_data = industry_data
    # store_python_obj(industryPchangeAllDataDic, dataDicFileName)
    # store_csv_data(industry_data, industryDataFileName)

    if industryPearsonrDataDic is None and concept is None:
        industryPearsonrDataDic = dict()
        for stockIndustryName in industryPchangeAllDataDic.keys():
            industryPchangeData = industryPchangeAllDataDic[stockIndustryName].fillna(value=0)
            industryPchangeData.drop('date', axis=1, inplace=True)
            industryPearsonrDataDic[stockIndustryName] = corr_matrix(industryPchangeData, similar_type=abupy.ECoreCorrType('pears'))
        ABuEnv.industryPearsonrDataDic = industryPearsonrDataDic
        # store_python_obj(industryPearsonrDataDic, industryPearsonrDataDicFileName)



def getIndustryDataLocal():
    '''行业数据规整，本地获取
                000895  000666  ...
    20190101    0.1     -0.2    ...
    20190102    6       8       ...
    ...
    '''
    # 根据天数进行判断，是否当前有可用数据。有则复用，否则新建。
    dataDicFileName = 'industryPchangeAllDataDic'
    industryPchangeAllDataDic = load_python_obj(dataDicFileName)
    industryDataFileName = 'industry_data'
    industry_data = load_csv_data(industryDataFileName)
    industryPearsonrDataDicFileName = 'industryPearsonrDataDic'
    industryPearsonrDataDic = load_python_obj(industryPearsonrDataDicFileName)

    if (industryPchangeAllDataDic is not None) and (industry_data is not None) and (industryPearsonrDataDic is not None):
        ABuEnv.industryPchangeAllDataDic = industryPchangeAllDataDic
        ABuEnv.industry_data = industry_data
        ABuEnv.industryPearsonrDataDic = industryPearsonrDataDic
        return
