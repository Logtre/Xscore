# -*- coding:utf-8 -*-

#import pandas as pd
#import numpy as np
from util import PrincipleComponentAnalysis
from file_io import FileIO
from draw_chart import DrawChart

#import seaborn as sns; sns.set()

class PCAProcess:

    def __init__(self):
        self.file_io = FileIO()
        self.pca = PrincipleComponentAnalysis()
        self.chart = DrawChart()

    def pca_process(self, file_path, dim_number):
        # ファイルオープン処理
        org_df = self.file_io.open_file_as_pandas(file_path,"utf-8")
        # 不要な顧客IDを削除
        Y = org_df['現金外支払合計'] + org_df['現金支払合計']
        print("Y's shape is {}".format(Y.shape))

        #df = org_df.drop(columns='顧客ID')
        X = org_df.drop(['顧客ID','現金支払合計','現金外支払合計'],axis=1)
        print("X's shape is {}".format(X.shape))

        rd = self.pca.fit(X, dim_number)
        df_rd = self.pca.fit_transform(X, dim_number)

        # グラフ描画
        self.chart.pca_scatter_plot(df_rd, Y)

        # 主成分の寄与率を出力する
        print('各次元の寄与率: {0}'.format(rd.explained_variance_ratio_))
        print('累積寄与率: {0}'.format(sum(rd.explained_variance_ratio_)))

        return df_rd, Y
