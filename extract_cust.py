# -*- coding:utf-8 -*-

from util.util import FindPrefectureCode, CountRecord
from util.file_io import FileIO

import datetime
import math

class ExtractCust:
    def __init__(self, in_path, in_char, out_path, out_char, id_path, shop_path, pref_path):
        self.pref_code = FindPrefectureCode()
        self.file_io = FileIO()
        self.in_path = in_path
        self.in_char = in_char
        self.out_path = out_path
        self.out_char = out_char
        self.id_path = id_path
        self.shop_path = shop_path
        self.pref_path = pref_path

    def extract(self):
        # 編集したいファイル（元ファイル）を開く
        file = self.file_io.open_file(self.in_path,"r",self.in_char)
        # 書き出し用のファイルを開く
        out_file = self.file_io.open_file(self.out_path,"w",self.out_char)
        id_file = self.file_io.open_file(self.id_path,"w",self.out_char)
        shop_file = self.file_io.open_file(self.shop_path,"w",self.out_char)
        pref_file = self.file_io.open_file(self.pref_path,"w",self.out_char)

        # 書き出し用ファイルのヘッダーを記述
        #out_file.write("顧客ID,担当店舗,生年月日,性別,携帯TEL,自宅TEL,携帯メール,PCメール,町域,職業\n")
        out_file.write("顧客ID,生年月日,性別,携帯TEL,自宅TEL,携帯メール,PCメール,職業\n")
        # id書き出し用ファルのヘッダーを記述
        id_file.write("顧客ID\n")
        # shop書き出し用ファイルのヘッダーを記述
        shop_file.write("顧客ID,担当店舗\n")
        # pref書き出し用ファイルのヘッダーを記述
        pref_file.write("顧客ID,町域\n")

        # 元ファイルのヘッダーをreadlineメソッドで１行飛ばす
        file.readline()
        # 元ファイルのレコード部分をreadlinesメソッドで全行を読み取る
        lines = file.readlines()

        # for文で1行ずつ取得
        for line in lines:
            # 改行コードをブランクに置換
            line = line.replace("\n","")
            # カンマ区切りでリストに変換する
            line = line.split(",")
            # 変換後のカンマ区切りの雛形を作り、変換処理した値を入れ込む
            row = "{},{},{},{},{},{},{},{}\n".format(
                line[3], #id
                #line[10], #shop
                line[15].replace("-",""), #birth
                line[16], #sex
                line[17], #moblie-num
                line[18], #tel-num
                line[19], #mobile-mail
                line[20], #pc-mail
                #self.pref_code.find_prefecture(line[22]), #address
                line[30], #job
                )
            id_row ="{}\n".format(
                line[3], #id
                )
            shop_row ="{},{}\n".format(
                line[3], #id
                line[10], #shop
            )
            pref_row ="{},{}\n".format(
                line[3], #id
                self.pref_code.find_prefecture(line[22]), #address
            )
            # 書き出し用のファイルに出力
            out_file.write(row)
            id_file.write(id_row)
            shop_file.write(shop_row)
            pref_file.write(pref_row)

        # ファイルを閉じる
        self.file_io.close_file(file)
        self.file_io.close_file(out_file)
        self.file_io.close_file(id_file)
        self.file_io.close_file(shop_file)
        self.file_io.close_file(pref_file)
