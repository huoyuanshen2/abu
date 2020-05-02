# -*- encoding:utf-8 -*-
"""量化技术分析工具图形可视化"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import ipywidgets as widgets

from ..WidgetBu.ABuWGBase import WidgetBase
from ..WidgetBu.ABuWGToolBase2 import WidgetToolSet2
from ..WidgetBu.ABuWGTLTool2 import WidgetTLTool2
from ..WidgetBu.ABuWGDATool import WidgetDATool
from ..WidgetBu.ABuWGSMTool import WidgetSMTool

__author__ = '阿布'
__weixin__ = 'abu_quant'


class WidgetQuantTool2(WidgetBase):
    """量化分析工具主界面"""

    def __init__(self):
        self.ts = WidgetToolSet2()
        self.tl = WidgetTLTool2(self.ts) #技术分析


