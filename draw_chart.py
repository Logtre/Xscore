# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab as plot

class DrawChart:

    def __init__(self):
        pass

    def scatter_plot(self, clf, X, Y):
        '''回帰直線を描画する'''
        # 散布図
        plt.scatter(X, Y)
        # 回帰直線
        plt.plt(X, clf.predict(X))

    def pca_scatter_plot(self, X, Y):
        '''次元削減した後の写像を描画する'''
        # 主成分をプロットする
        for label in np.unique(Y):
            plot.scatter(X[Y == label, 0],X[Y == label, 1])
        plot.title('principal component')
        plot.xlabel('pc1')
        plot.ylabel('pc2')
        # 描画
        plot.show()
