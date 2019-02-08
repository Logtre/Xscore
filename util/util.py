# -*- coding:utf-8 -*-
import csv
import pandas as pd
import numpy as np
#from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler

class Scaler:
    '''標準化・正規化処理を行うクラス'''
    def __init__(self):
        self.sc = StandardScaler()

    def standard_scaler(self, df, **kwargs):
        '''
        * 概要
        * 引数のDataFrameに対して、列ごとにaverage=0, mean=1の標準化を行なった上でDataFrameを返す
        * ------------------------------------------------------------------------------
        * Input
        *   1. df                      標準化を行うDataFrame
        *   2. kwargs["axis"]          0=行に対して標準化を行う
        *                              1=列に対して標準化を行う
        *   3. kwargs["data_type"]     標準化後のデータのデータ型
        * Output
        *   1. std_df                  DataFrame
        * ------------------------------------------------------------------------------
        '''
        ss = scale(df,axis=kwargs['axis'])
        # 標準化処理の返り値(numpy ndarray)を pandas dataframeに変換する
        std_df = pd.DataFrame(data=df, index=df.index, columns=df.columns, dtype=kwargs['data_type'])
        return std_df

    def sl_standard_scaler(self, df, **kwargs):
        ss = self.sc.fit_transform(df)
        # 標準化処理の返り値(numpy ndarray)を pandas dataframeに変換する
        std_df = pd.DataFrame(data=ss, index=df.index, columns=df.columns, dtype=kwargs['data_type'])
        return std_df


class ExtractColumns:
    '''指定した列の抽出・削除処理を行うクラス'''
    def __init__(self):
        pass

    def extract(self, df, id_col, **kwargs):
        '''指定列を抽出'''
        ex_col = id_col
        col_list = kwargs['extract_col']

        for col in col_list:
            target_col = df[col]
            ex_col = pd.concat([ex_col,target_col],axis=1)
        return ex_col

    def exclude(self, df, **kwargs):
        '''指定列を除外'''
        ex_col = df
        col_list = kwargs['exclude_col']

        for col in col_list:
            ex_col = ex_col.drop(col, axis=1)
        return ex_col


class PrincipleComponentAnalysis:
    '''次元圧縮処理を行うクラス'''
    def __init__(self):
        pass

    def fit_transform(self, pandas_obj, dim_number):
        '''次元を圧縮した上で、原関数を圧縮後関数に転写する'''
        pca = PCA(n_components=dim_number)
        print(pca.fit_transform(pandas_obj))
        return pca.fit_transform(pandas_obj)

    def fit(self, pandas_obj, dim_number):
        '''次元を圧縮する'''
        pca = PCA(n_components=dim_number)
        return pca.fit(pandas_obj)


class CategoryEncode:
    '''分類データを特徴量データに変換するクラス'''
    def __init__(self):
        #self.enc = OneHotEncoder()
        pass

    def transform_feature(self, pandas_obj, **kwargs):
        '''分類データを特徴量データに変換する'''
        # Encode列の指定
        col_list = kwargs['aggregate_col']

        # OneHotEncodeしたい列を指定
        ce_ohe = ce.OneHotEncoder(cols=col_list,handle_unknown='impute')

        pandas_ce_onehot = ce_ohe.fit_transform(pandas_obj)
        #print(pandas_ce_onehot)
        return pandas_ce_onehot

    def transform_feature_org(self, data):
        '''分類データを特徴量データに変換する'''
        self.enc.fit(data)
        feature_data = self.enc.transform(data).toarray()
        header = self.enc.get_feature_names()
        return header, feature_data

    def inverse_transform_feature(self, feature_data):
        '''特徴量データを分類データに逆変換する'''
        org_data = self.enc.inverse_transform(feature_data)
        return org_data

    def transform_label(self, data, column_name):
        '''分類データから特徴量データのラベル特定する'''
        # 返り値を格納するリストを定義
        name_list = []
        # ラベルエンコーダーを定義
        le = LabelEncoder()
        labels = le.fit_transform(data.astype(str))
        # 元の列名に特徴量データのラベルを結合する
        for label in le.classes_:
            label = column_name + '_' + label
            name_list.append(label)
        else:
            name_list.append(column_name + '_空欄')
        return name_list


class CountRecord:
    '''要素数（行数）をカウントするクラス'''
    def __init__(self):
        self.count = {}

    def count_record(self, df, col):
        '''要素毎のレコード数をカウントする'''
        vc = df[col].value_counts(dropna=False)
        #print(format(vc))
        return vc

    def group_size(self, df, **kwargs):
        '''複数項目でグルーピングし、要素数を返す'''
        #group = pandas_obj.groupby(kwargs['index_col'])[kwargs['aggregate_col']].size()
        group = df.groupby(kwargs['aggregate_col']).size()
        #print(format(group))
        return group

    def group_sum(self, df, **kwargs):
        '''複数項目でグルーピングし、指定した列の数値データを集計する'''
        group = df.groupby(kwargs['index_col'])[kwargs['aggregate_col']].sum()
        #print(format(group))
        return group

    def drop_duplicates(self, df, **kwargs):
        '''重複レコードを削除する'''
        group = df.drop_duplicates(subset=kwargs['index_col'])
        # 必要な列のみ抽出する
        matrix = group[kwargs['keep_list']]
        return matrix


class Binning:
    '''指定した列をグルーピング（ビニング）するクラス'''
    def __init__(self):
        pass

    def list_divide(self, df, list, label_name):
        '''リストで指定した範囲でdfを分割する'''
        ctgr_list = list
        ctgr_name = label_name
        return pd.cut(df, bins=ctgr_list, labels=ctgr_name)

    def quant_divide(self, df, quant, label_name):
        '''指定した数量で別れるようにdfを分割する'''
        quantity = quant
        ctgr_name = label_name
        return pd.qcut(df, q=quantity, labels=ctgr_name, duplicates='drop')


class FindPrefectureCode:
    '''文字列の住所から都道府県コードに変換するクラス'''
    def __init__(self):
        self.p_data = []
        self.c_data = []

        with open("./data/other/prefecture.csv","r",encoding="utf-8") as p_file:
            p_reader = csv.reader(p_file)
            for row in p_reader:
                self.p_data.append(row)

        with open("./data/other/city.csv","r",encoding="utf-8") as c_file:
            c_reader = csv.reader(c_file)
            for row in c_reader:
                self.c_data.append(row)

    def find_prefecture(self,addr):
        '''addressを都道府県名を鍵としてprefecture_codeに変換する'''
        # 都道府県名がaddressに含まれる場合はprefecture_codeを返す
        for pref in self.p_data:
            if pref[1] in addr:
                return pref[1]
        # 都道府県名でヒットしない場合、市区町村コードで検索する
        else:
            return self.find_city(addr)

    def find_city(self, addr):
        '''addressを市区町村名を鍵としてprefecture_codeに変換する
           ただし、検索する市区町村は政令指定都市までとする'''
        # 市区町村名がaddressに含まれる場合はprefecture_codeを返す
        for city in self.c_data:
            if city[2] in addr:
                return city[2]
        else:
            #print("other address is {}".format(addr))
            return 0
