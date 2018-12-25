# -*- coding:utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split

import sklearn.model_selection


class Test:
    def __init__(self):
        pass

    def make_train_test_data(self, X, Y, ratio):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=ratio)
        # NaNをゼロ埋め
        #X_train_zero = X_train.fillna(0)
        #X_test_zero = X_test.fillna(0)
        #Y_train_zero = Y_train.fillna(0)
        #Y_test_zero = Y_test.fillna(0)
        #return X_train_zero, X_test_zero, Y_train_zero, Y_test_zero
        return X_train, X_test, Y_train, Y_test
