# -*- coding:utf-8 -*-

import csv
from util import CountRecord
from file_io import FileIO
import pandas as pd

class ExtractLog:

    def __init__(self, in_path, in_char, stay_time_path, out_char, pv_sum_path, session_path):
        self.count_rec = CountRecord()
        self.file_io = FileIO()
        self.in_path = in_path
        self.in_char = in_char
        self.stay_time_path = stay_time_path
        self.out_char = out_char
        self.pv_sum_path = pv_sum_path
        self.session_path = session_path

    def extract(self):
        # ファイルオープン処理
        file = self.file_io.open_file_as_pandas(self.in_path,self.in_char)

        # 不要列を削除
        file = file.drop(['IPアドレス','メソッド','パス','HTTPバージョン','ファイル名','レスポンスバイト数','リファラ','ユーザーエージェント','レスポンスタイム'], axis=1)
        # timestamp列をdatetime表示
        file['アクセス日時_unix'] = pd.to_datetime(file['アクセス日時'])
        # アクセス日時の差(秒)を算出
        file['アクセス間隔'] = (file['アクセス日時_unix'].shift(-1) - file['アクセス日時_unix']).dt.seconds
        # 顧客IDの同一性を確認
        file['顧客ID同一当否'] = (file['顧客ID'].shift(-1) == file['顧客ID'])
        # IDが同一でないセルのアクセス間隔をゼロにする
        file.loc[~file['顧客ID同一当否'], 'アクセス間隔'] = 0
        # 同一セッションのアクセスであるフラグ
        file.loc[file['顧客ID同一当否'], 'セッションフラグ'] = 1

        # 総滞在時間
        stay_time = self.count_rec.group_sum(file, index_col='顧客ID', aggregate_col='アクセス間隔')
        # 閲覧ページ総数(集計処理)
        pv_sum = self.count_rec.count_record(file, '顧客ID')
        # セッション回数
        same_session = self.count_rec.group_sum(file, index_col='顧客ID', aggregate_col='セッションフラグ')

        # 書き出し処理
        #self.file_io.export_csv_from_pandas(file, './data/out/log.csv')
        self.file_io.export_csv_from_pandas(stay_time, self.stay_time_path)
        self.file_io.export_csv_from_pandas(pv_sum, self.pv_sum_path)
        self.file_io.export_csv_from_pandas(same_session, self.session_path)

        # ヘッダー付与のため再度ファイルオープン
        out_file1 = self.file_io.open_file_as_pandas(self.stay_time_path,self.out_char)
        out_file2 = self.file_io.open_file_as_pandas(self.pv_sum_path,self.out_char)
        out_file3 = self.file_io.open_file_as_pandas(self.session_path,self.out_char)
        # ヘッダー付与
        out_file1.columns = ['顧客ID','滞在時間']
        out_file2.columns = ['顧客ID','閲覧ページ総数']
        out_file3.columns = ['顧客ID','閲覧ページ数/セッション']
        # 書き出し処理
        self.file_io.export_csv_from_pandas(out_file1, self.stay_time_path)
        self.file_io.export_csv_from_pandas(out_file2, self.pv_sum_path)
        self.file_io.export_csv_from_pandas(out_file3, self.session_path)
