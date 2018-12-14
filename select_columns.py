# -*- coding:utf-8 -*-

import pandas as pd
# 独自ライブラリ
from file_io import FileIO
from util import ExtractColumns


class SelectColumns:

    def __init__(self, con_path, char_type):
        # 初期化
        self.file_io = FileIO()
        self.extract_col = ExtractColumns()
        self.con_path = con_path
        # ファイルオープン
        self.con = self.file_io.open_file_as_pandas(con_path, char_type)

    def select(self, **kwargs):
        # ターゲットリスト
        tg_list = kwargs['extract_col'] + ['現金外支払合計','現金支払合計']
        # ターゲット列を抽出
        target_col = self.extract_col.extract(self.con, self.con['顧客ID'], extract_col=tg_list)
        # ファイル書き込み
        self.file_io.export_csv_from_pandas(target_col, self.con_path)
