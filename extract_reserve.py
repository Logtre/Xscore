# -*- coding:utf-8 -*-

from util import CountRecord
from file_io import FileIO

class ExtractReserve:
    def __init__(self, in_path, in_char, out_path, out_char, reg_type_path):
        self.count_rec = CountRecord()
        self.file_io = FileIO()
        self.in_path = in_path
        self.in_char = in_char
        self.out_path = out_path
        self.out_char = out_char
        self.reg_type_path = reg_type_path

    def extract(self):
        # ファイルオープン処理
        file = self.file_io.open_file_as_pandas(self.in_path,self.in_char)
        # 集計処理1
        # 顧客ID, 状況, 指名区分を鍵としてレコード数を集計
        status = self.count_rec.group_size(file, index_col='顧客ID', aggregate_col=['顧客ID', '状況', '指名区分'])
        # 集計処理2
        # 顧客IDを鍵として認知媒体区分、登録区分を抽出
        register_type = self.count_rec.drop_duplicates(file, index_col='顧客ID', keep_list=['顧客ID','登録区分'])
        # 書き出し処理
        self.file_io.export_csv_from_pandas(status, self.out_path)
        self.file_io.export_csv_from_pandas(register_type, self.reg_type_path)

        # ヘッダー付与のため再度ファイルオープン
        out_file = self.file_io.open_file_as_pandas(self.out_path,self.out_char)
        # ヘッダー付与
        out_file.columns = ['顧客ID','状況', '指名区分', '予約回数']
        # 書き出し処理
        self.file_io.export_csv_from_pandas(out_file, self.out_path)
