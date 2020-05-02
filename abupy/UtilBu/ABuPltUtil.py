# -*- encoding:utf-8 -*-
"""
    画图工具模块
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import os

def generatePngName(stockCode):
    """
    生成图片保存名称。
    """
    from abupy.MarketBu.ABuMarketDrawing import K_SAVE_CACHE_PNG_ROOT
    from abupy.UtilBu import ABuDateUtil
    save_dir = os.path.join(K_SAVE_CACHE_PNG_ROOT, ABuDateUtil.current_str_date())
    png_dir = os.path.join(save_dir, stockCode)
    from abupy.UtilBu import ABuFileUtil
    ABuFileUtil.ensure_dir(png_dir)
    r_cnt = 0
    while True:
        png_name = '{}{}.png'.format(png_dir, '' if r_cnt == 0 else '-{}'.format(r_cnt))
        if not ABuFileUtil.file_exist(png_name):
            break
        r_cnt += 1
    return png_name

