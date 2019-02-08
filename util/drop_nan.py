# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import math

class DropNaN:

    def __init__(self):
        pass

    def drop_na_col(self, pd, rows, ratio):
        '''
        欠損値が所定の個数含まれる列を削除する関数
            @param  pd      評価対象データセット
            @param  rows    削除基準となる欠損値の数
            @param  ratio   削除基準の何割をボーダーラインにするかの係数
                            rows * ratioでボーダーラインとなる欠損値の個数を定義する
        '''
        barrier = rows * ratio
        return pd.dropna(thresh = barrier, axis=1)

    def drop_minus_row(self, pd, target):
        '''
        マイナスの値をとる行を削除する関数
            @param  pd      評価対象データセット
            @param  target  評価対象列（この列のマイナス値が含まれる行を削除する）
        '''
        return pd.drop(pd[pd[target] <= 0].index)
        self.cust.drop(self.cust[self.cust['生年月日'].str.match('\*', na=False)].index)
