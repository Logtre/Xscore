# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import math

class DropNaN:

    def __init__(self):
        pass

    def drop_na_col(self, pd, rows, ratio):
        barrier = rows * ratio
        return pd.dropna(thresh = barrier, axis=1)

    def drop_minus_row(self, pd, target):
        return pd.drop(pd[pd[target] <= 0].index)
        self.cust.drop(self.cust[self.cust['生年月日'].str.match('\*', na=False)].index)
