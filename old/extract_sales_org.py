# -*- coding:utf-8 -*-
import pandas as pd

# 独自ライブラリ
from util import CountRecord
from file_io import FileIO

class ExtractSales:
    def __init__(self, in_path, in_char, payment_path, out_char, cust_attr_path, product_attr_path):
        self.count_rec = CountRecord()
        self.file_io = FileIO()
        self.in_path = in_path
        self.in_char = in_char
        self.payment_path = payment_path
        self.out_char = out_char
        self.cust_attr_path = cust_attr_path
        self.product_attr_path = product_attr_path

    def extract(self):
        # ファイルオープン処理
        file = self.file_io.open_file_as_pandas(self.in_path,self.in_char)

        # 前処理：個別商品の行を売上の行に統合
        sales_file = file.query('明細コード == 1')

        # 集計処理1.1：顧客IDごとの支払情報を集計
        cust_payment1 = self.count_rec.group_sum(sales_file, index_col='顧客ID', aggregate_col=['顧客ID','施術時間','現金支払合計','現金外支払合計','未収金前回残高'])
        # 集計処理1.2：顧客IDごとの個別商品情報を集計
        cust_payment2 = self.count_rec.group_sum(file, index_col='顧客ID', aggregate_col=['顧客ID','売上単価','数量'])
        # 集計処理1.3：支払情報をマージ
        cust_payment = pd.merge(cust_payment1, cust_payment2, on='顧客ID', how='left')

        # 集計処理2.1：顧客IDごとの属性情報を集計
        ex_id = sales_file['顧客ID']
        ex_nominate = sales_file['指名回数']
        ex_course = sales_file['コース受諾回数']
        ex_card = sales_file['紹介カード受渡回数']
        ex_reception = sales_file['治療送客回数']
        ex_director = sales_file['院長挨拶回数']

        # マージ
        cust_attr = pd.concat([ex_id, ex_nominate],axis=1)
        cust_attr = pd.concat([cust_attr, ex_course],axis=1)
        cust_attr = pd.concat([cust_attr, ex_card],axis=1)
        cust_attr = pd.concat([cust_attr, ex_reception],axis=1)
        cust_attr = pd.concat([cust_attr, ex_director],axis=1)
        #cust_attr = self.cont_rec.group_size(sales_file, index_col='顧客ID', keep_list=['顧客ID','指名回数','コース受託回数','紹介カード受渡回数','治療送客回数','院長挨拶回数'])

        # 集計処理2.2：顧客IDごとの個別商品属性情報を集計
        ex_id_product = file['顧客ID']
        ex_product_code = file['商品コード']
        ex_sales_type = file['売上区分']
        ex_product_type = file['商品区分']
        # マージ
        product_attr = pd.concat([ex_id_product, ex_product_code],axis=1)
        product_attr = pd.concat([product_attr, ex_sales_type],axis=1)
        product_attr = pd.concat([product_attr, ex_product_type],axis=1)
        #product_attr = file['顧客ID','商品コード','売上区分','商品区分']
        #product_attr = self.cont_rec.group_size(file, index_col='顧客ID', keep_list=['顧客ID','商品コード','売上区分','商品区分'])

        # 書き出し処理
        self.file_io.export_csv_from_pandas(cust_payment, self.payment_path)
        self.file_io.export_csv_from_pandas(cust_attr, self.cust_attr_path)
        self.file_io.export_csv_from_pandas(product_attr, self.product_attr_path)
