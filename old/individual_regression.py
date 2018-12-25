# -*- coding:utf-8 -*-
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

import pandas as pd
import numpy as np

import json

# 独自ライブラリ
#from util import PrincipleComponentAnalysis
from file_io import FileIO
from drop_nan import DropNaN
#from pca import PCAProcess
#from draw_chart import DrawChart
from test_util import Test
from individual_test.regression_test import IndividualTest


# グラフ描画
#from matplotlib import pylab as plt
#import seaborn as sns; sns.set()

class IndRegression:

    def __init__(self):
        #self.lr = LinearRegression()
        self.file_io = FileIO()
        #self.pca = PCAProcess()
        #self.chart = DrawChart()
        self.test = Test()
        self.individual = IndividualTest()
        self.sc = StandardScaler()
        self.ms = MinMaxScaler()
        self.drop_na = DropNaN()

        self.droplist = []
        with open('droplist.txt') as f:
            self.droplist = [s.strip() for s in f.readlines()]

    def regression(self, in_path, out_path):
        # ファイルオープン処理
        org_df = self.file_io.open_file_as_pandas(in_path,"utf-8")

        '''
        # 目的変数
        org_df['支払合計'] = org_df['現金外支払合計'] + org_df['現金支払合計']
        # 不要な説明変数削除
        org_df = org_df.drop(['現金外支払合計', '現金支払合計'],axis=1)
        # 売上関連説明変数削除
        org_df = org_df.drop(self.droplist,axis=1)
        # 目的の下限を設定
        org_df = org_df.drop(org_df[org_df['支払合計']<=0].index)
        # 目的変数の上限を設定
        org_df = org_df.drop(org_df[org_df['支払合計']>=40000].index)
        '''
        # 年齢の下限を設定
        org_df = org_df.drop(org_df[org_df['生年月日']<=20].index)
        # 年齢の上限を設定
        org_df = org_df.drop(org_df[org_df['生年月日']>=50].index)
        # 閲覧回数の下限を設定
        org_df = org_df.drop(org_df[org_df['閲覧ページ総数']<=0].index)
        # 閲覧回数の上限を設定
        org_df = org_df.drop(org_df[org_df['閲覧ページ総数']>=100].index)
        # スコア=0を削除
        org_df = org_df.drop(org_df[org_df['スコア']<=0].index)
        '''
        # 欠損値が多すぎる列を削除
        #org_df = org_df.drop(['売上単価'],axis=1)
        # 目的変数が欠損値の行を削除
        org_df = org_df.dropna(subset=['支払合計'])
        '''
        # 不要列削除
        #org_df = org_df.drop(['Unnamed: 0', '顧客ID'], axis=1)
        org_df = org_df.drop(['顧客ID'],axis=1)
        org_df = org_df[org_df.columns.drop(list(org_df.filter(regex='Unnamed:')))]
        # 欠損値が70%以上の列を削除
        #org_df = self.drop_na.drop_na_col(org_df, len(org_df), 0.7)
        #print('\n rows of org_df is:')
        #print(len(org_df))
        #print(type(len(org_df)))
        # 欠損値をゼロうめ
        #org_df = org_df.fillna(0)

        # 説明変数Y
        #Y = org_df['支払合計']
        Y = org_df['スコア']

        # 10等分
        #bin_Y = pd.cut(org_Y, 2, labels=False)
        #print(bin_Y)

        # 目的変数X
        #X = org_df.drop(['支払合計'],axis=1)
        X = org_df.drop(['商品コード','売上単価','数量','売上','明細ID','スコア'],axis=1)
        # 属性情報削除
        X = X.drop(['キャンセル回数','コンタクト回数','問い合わせ回数'],axis=1)
        X = X[X.columns.drop(list(org_df.filter(regex='施術時間')))]
        X = X[X.columns.drop(list(org_df.filter(regex='指名回数')))]
        #X = X[X.columns.drop(list(org_df.filter(regex='コース受諾回数')))]
        X = X[X.columns.drop(list(org_df.filter(regex='紹介カード受渡回数')))]
        X = X[X.columns.drop(list(org_df.filter(regex='治療送客回数')))]
        X = X[X.columns.drop(list(org_df.filter(regex='院長挨拶回数')))]
        X = X[X.columns.drop(list(org_df.filter(regex='性別')))]
        X = X[X.columns.drop(list(org_df.filter(regex='携帯TEL')))]
        X = X[X.columns.drop(list(org_df.filter(regex='自宅TEL')))]
        X = X[X.columns.drop(list(org_df.filter(regex='携帯メール')))]
        X = X[X.columns.drop(list(org_df.filter(regex='PCメール')))]
        X = X[X.columns.drop(list(org_df.filter(regex='職業')))]
        X = X[X.columns.drop(list(org_df.filter(regex='登録区分')))]
        

        # 欠損値をゼロうめ
        Y = Y.fillna(0)
        X = X.fillna(0)

        # 個別テスト
        self.individual.lin_reg(X, Y, 0.3, X.columns, out_path)
