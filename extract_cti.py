# -*- coding:utf-8 -*-

import csv
from util.util import CountRecord
from util.file_io import FileIO

class ExtractCti:
    def __init__(self, in_path, in_char, out_path, out_char):
        self.count_rec = CountRecord()
        self.file_io = FileIO()
        self.in_path = in_path
        self.in_char = in_char
        self.out_path = out_path
        self.out_char = out_char

    def extract(self):
        # ファイルオープン処理
        file = self.file_io.open_file_as_pandas(self.in_path,self.in_char)
        # 集計処理
        vc = self.count_rec.count_record(file, '顧客ID')
        # 書き出し処理
        self.file_io.export_csv_from_pandas(vc, self.out_path)

        # ヘッダー付与のため再度ファイルオープン
        out_file = self.file_io.open_file_as_pandas(self.out_path,self.out_char)
        # ヘッダー付与
        out_file.columns = ['顧客ID','問い合わせ回数']
        # 書き出し処理
        self.file_io.export_csv_from_pandas(out_file, self.out_path)
