# -*- encoding:utf-8 -*-
"""
    示例仓位管理：atr仓位管理模块
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from .ABuPositionBase import AbuPositionBase

__author__ = '阿布'
__weixin__ = 'abu_quant'

"""
    默认0.1即10% 外部可通过如：abupy.beta.atr.g_atr_pos_base = 0.01修改仓位基础配比
    需要注意外部其它自定义仓位管理类不要随意使用模块全局变量，AbuAtrPosition特殊因为注册
    在ABuEnvProcess中在多进程启动时拷贝了模块全局设置内存
"""
g_atr_pos_base = 0.1


class AbuRatePosition(AbuPositionBase):
    """按总金额固定比例买入_仓位管理类"""

    base_rate = 0.1  # 每次买入费用占全部初始金额比例。

    def fit_position(self, factor_object):
        """
        fit_position计算的结果是买入多少个单位（股，手，顿，合约）
        计算：（常数价格 ／ 买入价格）＊ 当天交易日atr21
        :param factor_object: ABuFactorBuyBases实例对象
        :return: 买入多少个单位（股，手，顿，合约）
        """
        # 结果是买入多少个单位（股，手，顿，合约）
        return self.read_cash * self.base_rate / self.bp

    def _init_self(self, **kwargs):
        """atr仓位控制管理类初始化设置"""
        self.base_rate = kwargs.pop('base_rate', AbuRatePosition.base_rate) #基础比例
