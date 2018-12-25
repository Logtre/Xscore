# -*- coding:utf-8 -*-
from sklearn import tree

import pandas as pd
import numpy as np

# 独自ライブラリ
#from util import PrincipleComponentAnalysis
from file_io import FileIO
#from pca import PCAProcess
#from draw_chart import DrawChart
from test_util import Test

# グラフ描画
#from matplotlib import pylab as plt
#import seaborn as sns; sns.set()

class DecisionTree:

    def __init__(self, path, depth):
        self.clf = tree.DecisionTreeClassifier(max_depth=3)
        self.file_io = FileIO()
        #self.pca = PCAProcess()
        #self.chart = DrawChart()
        self.test = Test()
        self.file_path = path

    def analyze(self):
        # ファイルオープン処理
        org_df = self.file_io.open_file_as_pandas(self.file_path,"utf-8")

        # 目的変数Xと説明変数Y
        Y = org_df['現金外支払合計'] + org_df['現金支払合計']
        X = org_df.drop(['顧客ID','現金支払合計','現金外支払合計'],axis=1)
        # Xの各列を正規化
        X_normal = X.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))

        # トレーニングデータとテストデータに分割(30%)
        X_train, X_test, Y_train, Y_test = self.test.make_train_test_data(X_normal, Y, 0.3)
        print(X_train.head())
        print("--- X_train's shape ---\n {}\n".format(X_train.shape))
        print(X_test.head())
        print("--- X_test's shape ---\n {}\n".format(X_test.shape))
        print(Y_train.head())
        print("--- Y_train's shape ---\n {}\n".format(Y.shape))
        print(Y_test.head())
        print("--- Y_test's shape ---\n {}\n".format(Y.shape))


        # 分析を実施
        predicted = self.clf.fit(X_train, Y_train)
        # 識別率を確認
        ratio = sum(predicted == Y_train) / len(Y_train)

        # 精度を算出
        # トレーニングデータ
        print(" --- train score ---\n {}\n".format(self.lr.score(X_train,Y_train)))
        # テストデータ
        print(" --- test score ---\n {}\n".format(self.lr.score(X_train,Y_train)))

        return self.lr.score(X_train,Y_train), self.lr.score(X_train,Y_train)
