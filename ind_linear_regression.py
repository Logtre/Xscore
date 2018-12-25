# -*- coding:utf-8 -*-

import sys
import io

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import pandas as pd

# 独自ライブラリ
from util.file_io import FileIO
from util.test_util import Test
from util.draw_chart2 import DrawChart2

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

class IndividualTest:
    def __init__(self):
        self.test = Test()
        self.file_io = FileIO()
        self.lr = LinearRegression(normalize=True)
        self.br = BayesianRidge()
        #self.svr_lin = SVR(kernel='linear', C=1e5)
        self.svr_poly = SVR(kernel='poly', C=1e5, degree=2)
        self.svr_rbf = SVR(kernel='rbf', C=5e4, gamma='scale')
        self.svr_sig = SVR(kernel='sigmoid', C=1e3)
        #self.gridsearch = GridSearchCV(SVR(kernel='rbf'), scoring="r2", return_train_score=True)
        self.sc = StandardScaler()
        self.ms = MinMaxScaler()
        self.chart = DrawChart2()

    def lin_reg(self, X, Y, train_test_ratio, col_list, out_path):

        # 空のDataFrameを作成
        df = pd.DataFrame(index=['coefficient','intercept','train_score','test_score'], columns=[])
        #print(df.head())

        for col in col_list:
            s_X = pd.DataFrame(X[col])
            s_Y = Y

            # トレーニングデータとテストデータに分割(30%)
            s_X_train, s_X_test, s_Y_train, s_Y_test = self.test.make_train_test_data(s_X, s_Y, train_test_ratio)

            # 列ごとに単回帰分析
            self.lr.fit(s_X_train, s_Y_train)

            # 偏回帰係数
            coef = self.lr.coef_

            # 切片 (誤差)
            intercept = self.lr.intercept_

            # トレーニングスコア
            train_score = self.lr.score(s_X_train,s_Y_train)

            # テストスコア
            test_score = self.lr.score(s_X_test,s_Y_test)

            # DataFrameに追加
            df[col] = [coef, intercept, train_score, test_score]

            # 回帰曲線
            lin_pred = self.lr.predict(s_X_test)

            plt.plot(s_X_test, s_Y_test, 'bo-', s_X_test, lin_pred, 'go-')
            plt.show()

            #if col in ['売上単価','コース受諾回数_なし','数量','施術時間','指名回数_あり','治療送客回数_あり','治療送客回数_なし']:
                # グラフ描画
                #self.chart.draw(self.lr, s_X_test, s_Y_test, col, 'score is {}'.format(test_score))

        # csvファイルに書き出し
        self.file_io.export_csv_from_pandas(df, out_path)

    def bayesian_reg(self, X, Y, train_test_ratio, col_list, out_path):

        # 空のDataFrameを作成
        df = pd.DataFrame(index=['coefficient','intercept','train_score','test_score'], columns=[])
        #print(df.head())

        for col in col_list:
            s_X = pd.DataFrame(X[col])
            s_Y = Y

            # トレーニングデータとテストデータに分割(30%)
            s_X_train, s_X_test, s_Y_train, s_Y_test = self.test.make_train_test_data(s_X, s_Y, train_test_ratio)

            # 列ごとに単回帰分析
            self.br.fit(s_X_train, s_Y_train)

            # 偏回帰係数
            coef = self.br.coef_

            # 切片 (誤差)
            intercept = self.br.intercept_

            # トレーニングスコア
            train_score = self.br.score(s_X_train,s_Y_train)

            # テストスコア
            test_score = self.br.score(s_X_test,s_Y_test)

            # DataFrameに追加
            df[col] = [coef, intercept, train_score, test_score]

            if col in ['売上単価','コース受諾回数_なし','数量','施術時間','指名回数_あり','治療送客回数_あり','治療送客回数_なし']:
                # グラフ描画
                self.chart.draw(self.br, s_X_test, s_Y_test, col, 'score is {}'.format(test_score))

        # csvファイルに書き出し
        self.file_io.export_csv_from_pandas(df, out_path)

    def svr_rbf_reg(self, X, Y, train_test_ratio, col_list, out_path):

        # 空のDataFrameを作成
        df = pd.DataFrame(index=['coefficient','suport_vector','intercept','train_score','test_score'], columns=[])
        #print(df.head())

        for col in col_list:
            s_X = pd.DataFrame(X[col])
            s_Y = Y

            # トレーニングデータとテストデータに分割(30%)
            s_X_train, s_X_test, s_Y_train, s_Y_test = self.test.make_train_test_data(s_X, s_Y, train_test_ratio)

            # 列ごとに回帰分析
            #self.svr_lin.fit(s_X_train, s_Y_train)
            #self.svr_poly.fit(s_X_train, s_Y_train)
            self.svr_rbf.fit(s_X_train, s_Y_train)
            #self.gridsearch.fit(s_X_train, s_Y_train)

            # 偏回帰係数
            coef = self.svr_rbf.dual_coef_

            # サポートベクトル
            support_vec = self.svr_rbf.support_vectors_

            # 切片 (誤差)
            intercept = self.svr_rbf.intercept_

            # 精度
            train_score = self.svr_rbf.score(s_X_train, s_Y_train)
            test_score = self.svr_rbf.score(s_X_test, s_Y_test)

            # DataFrameに追加
            df[col] = [coef, support_vec, intercept, train_score, test_score]

            #lin_pred = self.svr_lin.predict(s_X_test)
            #poly_pred = self.svr_poly.predict(s_X_test)
            rbf_pred = self.svr_rbf.predict(s_X_test)

            plt.plot(s_X_test, s_Y_test, 'bo-', s_X_test, rbf_pred, 'go-')
            plt.show()

            if col in ['生年月日']:
                #plt.plot(s_X_test, s_Y_test, 'bo-', s_X_test, lin_pred, 'ro-')
                #plt.plot(s_X_test, s_Y_test, 'bo-', s_X_test, poly_pred, 'yo-')
                plt.plot(s_X_test, s_Y_test, 'bo-', s_X_test, rbf_pred, 'go-')
                plt.show()

        # csvファイルに書き出し
        self.file_io.export_csv_from_pandas(df, out_path)

    def svr_poly_reg(self, X, Y, train_test_ratio, col_list, out_path):

        # 空のDataFrameを作成
        df = pd.DataFrame(index=['coefficient','suport_vector','intercept','train_score','test_score'], columns=[])
        #print(df.head())

        for col in col_list:
            s_X = pd.DataFrame(X[col])
            s_Y = Y

            # トレーニングデータとテストデータに分割(30%)
            s_X_train, s_X_test, s_Y_train, s_Y_test = self.test.make_train_test_data(s_X, s_Y, train_test_ratio)

            # 列ごとに回帰分析
            self.svr_poly.fit(s_X_train, s_Y_train)

            # 偏回帰係数
            coef = self.svr_poly.dual_coef_

            # サポートベクトル
            support_vec = self.svr_poly.support_vectors_

            # 切片 (誤差)
            intercept = self.svr_poly.intercept_

            # 精度
            train_score = self.svr_poly.score(s_X_train, s_Y_train)
            test_score = self.svr_poly.score(s_X_test, s_Y_test)

            # DataFrameに追加
            df[col] = [coef, support_vec, intercept, train_score, test_score]

            #lin_pred = self.svr_lin.predict(s_X_test)
            #poly_pred = self.svr_poly.predict(s_X_test)
            rbf_pred = self.svr_poly.predict(s_X_test)


            if col in ['生年月日']:
                #plt.plot(s_X_test, s_Y_test, 'bo-', s_X_test, lin_pred, 'ro-')
                plt.plot(s_X_test, s_Y_test, 'bo-', s_X_test, poly_pred, 'yo-')
                #plt.plot(s_X_test, s_Y_test, 'bo-', s_X_test, rbf_pred, 'go-')
                plt.show()

        # csvファイルに書き出し
        self.file_io.export_csv_from_pandas(df, out_path)

    def svr_sig_reg(self, X, Y, train_test_ratio, col_list, out_path):

        # 空のDataFrameを作成
        df = pd.DataFrame(index=['coefficient','suport_vector','intercept','train_score','test_score'], columns=[])
        #print(df.head())

        for col in col_list:
            s_X = pd.DataFrame(X[col])
            s_Y = Y

            # トレーニングデータとテストデータに分割(30%)
            s_X_train, s_X_test, s_Y_train, s_Y_test = self.test.make_train_test_data(s_X, s_Y, train_test_ratio)

            # 列ごとに回帰分析
            self.svr_sig.fit(s_X_train, s_Y_train)

            # 偏回帰係数
            coef = self.svr_sig.dual_coef_

            # サポートベクトル
            support_vec = self.svr_sig.support_vectors_

            # 切片 (誤差)
            intercept = self.svr_sig.intercept_

            # 精度
            train_score = self.svr_sig.score(s_X_train, s_Y_train)
            test_score = self.svr_sig.score(s_X_test, s_Y_test)

            # DataFrameに追加
            df[col] = [coef, support_vec, intercept, train_score, test_score]

            sig_pred = self.svr_sig.predict(s_X_test)

            plt.plot(s_X_test, s_Y_test, 'bo-', s_X_test, sig_pred, 'go-')
            plt.show()

            if col in ['生年月日','閲覧ページ総数','閲覧ページ数/セッション','滞在時間']:
                plt.plot(s_X_test, s_Y_test, 'bo-', s_X_test, sig_pred, 'go-')
                plt.show()

        # csvファイルに書き出し
        self.file_io.export_csv_from_pandas(df, inifile.get('regression', 'ind_path'))
