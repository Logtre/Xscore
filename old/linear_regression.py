# -*- coding:utf-8 -*-
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np

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

class LinRegression:

    def __init__(self):
        self.lr = LinearRegression()
        self.file_io = FileIO()
        #self.pca = PCAProcess()
        #self.chart = DrawChart()
        self.test = Test()
        self.individual = IndividualTest()
        self.sc = StandardScaler()
        self.ms = MinMaxScaler()
        self.drop_na = DropNaN()

    def regression(self, in_path, out_path):
        # ファイルオープン処理
        org_df = self.file_io.open_file_as_pandas(in_path,"utf-8")
        feat_shop = self.file_io.open_file_as_pandas('./data/out/feat_shop.csv','utf-8')
        feat_pref = self.file_io.open_file_as_pandas('./data/out/feat_pref.csv','utf-8')

        '''
        # 目的変数
        org_df['支払合計'] = org_df['現金外支払合計'] + org_df['現金支払合計']
        # 不要な説明変数削除
        org_df = org_df.drop(['現金外支払合計', '現金支払合計'],axis=1)
        # 目的変数がゼロ以下の行を削除
        org_df = org_df.drop(org_df[org_df['支払合計']==0].index)
        # 欠損値が多すぎる列を削除
        #org_df = org_df.drop(['売上単価'],axis=1)
        # 目的変数が欠損値の行を削除
        org_df = org_df.dropna(subset=['支払合計'])
        '''
        # shop追加
        #org_df = pd.merge(org_df, feat_shop, on='顧客ID',how='left')
        org_df = pd.merge(org_df, feat_pref, on='顧客ID',how='left')
        org_df = org_df.drop(['Unnamed: 0_x','Unnamed: 0_y'],axis=1)
        org_df = org_df[org_df.columns.drop(list(org_df.filter(regex='Unnamed:')))]
        # スコア=0を削除
        org_df = org_df.drop(org_df[org_df['スコア']<=0].index)
        # 不要列削除
        #org_df = org_df.drop(['Unnamed: 0', '顧客ID'], axis=1)
        org_df = org_df.drop(['顧客ID'],axis=1)
        #org_df = org_df[org_df.columns.drop(list(org_df.filter(regex='Unnamed:')))]
        #org_df = org_df.columns.drop(org_df.columns.str.contains('Unnamed:'))
        # 欠損値が70%以上の列を削除
        #org_df = self.drop_na.drop_na_col(org_df, len(org_df), 0.7)
        #print('\n rows of org_df is:')
        #print(len(org_df))
        #print(type(len(org_df)))
        # 欠損値をゼロうめ
        org_df = org_df.fillna(0)

        # 目的変数Xと説明変数Y
        #Y = org_df['売上']
        Y = org_df['スコア']
        #X = org_df.drop(['支払合計'],axis=1)
        X = org_df.drop(['商品コード','売上単価','数量','売上','明細ID','スコア'],axis=1)
        X = X.drop(['キャンセル回数','コンタクト回数','問い合わせ回数'],axis=1)
        X = X.drop(['治療送客回数_あり','治療送客回数_なし','院長挨拶回数_あり','院長挨拶回数_なし','紹介カード受渡回数_あり','紹介カード受渡回数_なし','携帯TEL_有','携帯メール_有','性別_女','性別_男','自宅TEL_有','PCメール_有'],axis=1)
        #X = X.drop(['職業_学生','職業_会社員','職業_主婦','職業_自営業','職業_その他','職業_パート・アルバイト'],axis=1)
        X = X.drop(['登録区分_HP','登録区分_店舗','登録区分_CC'],axis=1)
        #X = X.drop(['生年月日','滞在時間','閲覧ページ総数','閲覧ページ数/セッション'],axis=1)
        X = X[X.columns.drop(list(org_df.filter(regex='_nan')))]
        #X = X[X.columns.drop(list(org_df.filter(regex='_なし')))]
        X = X[X.columns.drop(list(org_df.filter(regex='_空欄')))]
        X = X[X.columns.drop(list(org_df.filter(regex='_無')))]
        X = X[X.columns.drop(list(org_df.filter(regex='_削除')))]
        X = X[X.columns.drop(list(org_df.filter(regex='施術時間')))]


        '''
        X = X[X.columns.drop(list(org_df.filter(regex='キャンセル回数')))]
        X = X[X.columns.drop(list(org_df.filter(regex='コンタクト回数')))]
        X = X[X.columns.drop(list(org_df.filter(regex='問い合わせ回数')))]
        X = X[X.columns.drop(list(org_df.filter(regex='滞在時間')))]
        X = X[X.columns.drop(list(org_df.filter(regex='閲覧ページ総数')))]
        X = X[X.columns.drop(list(org_df.filter(regex='閲覧ページ数/セッション')))]
        '''
        # 標準化
        #std_Y = pd.DataFrame(self.sc.fit_transform(Y))
        #std_Y.columns = Y.columns
        #std_X = pd.DataFrame(self.sc.fit_transform(X))
        #std_X.columns = X.columns

        # 正規化
        #norm_Y = pd.DataFrame(self.ms.fit_transform(Y))
        #norm_Y.columns = Y.columns
        #norm_X = pd.DataFrame(self.ms.fit_transform(X))
        #norm_X.columns = X.columns
        #self.file_io.export_csv_from_pandas(X, './data/out/X.csv')

        # トレーニングデータとテストデータに分割(30%)
        X_train, X_test, Y_train, Y_test = self.test.make_train_test_data(X, Y, 0.3)
        print(X_train.head())
        print("--- X_train's shape ---\n {}\n".format(X_train.shape))
        print(X_test.head())
        print("--- X_test's shape ---\n {}\n".format(X_test.shape))
        print(Y_train.head())
        print("--- Y_train's shape ---\n {}\n".format(Y_train.shape))
        print(Y_test.head())
        print("--- Y_test's shape ---\n {}\n".format(Y_test.shape))


        # 重回帰分析を実施
        self.lr.fit(X_train, Y_train)
        # 偏回帰係数
        print(pd.DataFrame({"Name":X.columns,
                            "Coefficients":self.lr.coef_}).sort_values(by='Coefficients') )
        # 切片 (誤差)
        print(self.lr.intercept_)

        # pandasファイル作成
        org_pd = pd.DataFrame({"Name":X.columns,
                            "Coefficients":self.lr.coef_})
        # ファイルアウトプット
        self.file_io.export_csv_from_pandas(org_pd, "./data/out/linear_regression.csv")

        # 精度を算出
        # トレーニングデータ
        print(" --- train score ---\n {}\n".format(self.lr.score(X_train,Y_train)))
        # テストデータ
        print(" --- test score ---\n {}\n".format(self.lr.score(X_test,Y_test)))

        return self.lr.score(X_train,Y_train), self.lr.score(X_test,Y_test)
