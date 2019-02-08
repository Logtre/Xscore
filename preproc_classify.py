# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from util.util import CategoryEncode, ExtractColumns, Binning, CountRecord, Scaler
from util.file_io import FileIO

import datetime
import math


class PreprocClassify:
    def __init__(
        self,
        id_path,
        con_path,
        char_type):

        self.file_io = FileIO()
        self.encode = CategoryEncode()
        self.count_rec = CountRecord()
        self.extract_col = ExtractColumns()
        self.bin = Binning()
        self.ss = Scaler()
        # ファイルオープン
        self.id = self.file_io.open_file_as_pandas(id_path,char_type)
        self.con = self.file_io.open_file_as_pandas(con_path,char_type)



    def make_class_data(self, out_path):
        '''目的変数を識別し、分析対象ファイルにマージする'''

        # 売上<=0を削除
        #org_df = self.con.drop(self.con[self.con['売上']<=0].index)

        # 目的変数列を抽出
        cust_attr_col_list =[] # 抽出列リストを初期化
        cust_attr_tg_list = ['売上'] # 抽出列リストに目的変数列を追加
        cust_con_col = self.extract_col.extract(self.con, self.con['顧客ID'], extract_col=cust_attr_tg_list)

        # 不要な顧客ID列を削除
        cust_con_col = cust_con_col.drop(['顧客ID'],axis=1)

        # 欠損値をゼロうめ
        cust_con_col = cust_con_col.fillna(0)

        # 抽出した目的変数列に対して、標準化（平均0, 分散1）処理を行う
        std_cust_con_col = self.ss.sl_standard_scaler(cust_con_col, data_type='float')

        # 標準化された目的変数列を平均より上か下かで識別
        type_bins = [-1, 0, 1] # 範囲:(-1,1), 0で分割する
        type_bin_label_list = [0,1] # 0より小: low, 0より大: high
        type_col = self.bin.list_divide(std_cust_con_col['売上'],type_bins,type_bin_label_list) # 分類用データの生成
        type_df = pd.DataFrame(data=type_col, index=std_cust_con_col.index) # 分類用データ（numpy）をdataframeに変更
        type_df.columns = ['クラス'] # dataframeのカラム名を変更

        # 分類用データを既存の分析用データにマージ
        type_df = pd.concat([self.id, type_df],axis=1) # id列とtype_dfを連結
        con = pd.merge(self.con, type_df, on='顧客ID',how='left') # 既存dataframeとtype_dfを連結

        # 書き出し処理
        self.file_io.export_csv_from_pandas(con, out_path)
