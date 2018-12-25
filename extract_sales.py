# -*- coding:utf-8 -*-
import pandas as pd

# 独自ライブラリ
from util.util import CountRecord
from util.file_io import FileIO

class ExtractSales:
    def __init__(self, in_path, in_char, payment_path, out_char, cust_attr_path, target_attr_path, average_attr_path):
        self.count_rec = CountRecord()
        self.file_io = FileIO()
        self.in_path = in_path
        self.in_char = in_char
        self.payment_path = payment_path
        self.out_char = out_char
        self.cust_attr_path = cust_attr_path
        self.target_attr_path = target_attr_path
        self.average_attr_path = average_attr_path

    def extract(self):
        # ファイルオープン処理
        file = self.file_io.open_file_as_pandas(self.in_path,self.in_char)

        # 顧客属性前処理：顧客属性取得のため、個別商品の行を売上の行に統合
        sales_file = file.query('明細コード == 1')

        # 集計処理：顧客IDごとの支払情報を集計
        cust_payment = self.count_rec.group_sum(sales_file, index_col='顧客ID', aggregate_col=['顧客ID','施術時間'])

        # 顧客属性集計処理：顧客IDごとの属性情報を集計
        ex_id = sales_file['顧客ID']
        ex_nominate = sales_file['指名回数']
        ex_course = sales_file['コース受諾回数']
        ex_card = sales_file['紹介カード受渡回数']
        ex_reception = sales_file['治療送客回数']
        ex_director = sales_file['院長挨拶回数']
        # 追加顧客属性
        #ex_branch = sales_file['店舗']
        #ex_accosiate = sales_file['担当者']

        # マージ
        cust_attr = pd.concat([ex_id, ex_nominate],axis=1)
        cust_attr = pd.concat([cust_attr, ex_course],axis=1)
        cust_attr = pd.concat([cust_attr, ex_card],axis=1)
        cust_attr = pd.concat([cust_attr, ex_reception],axis=1)
        cust_attr = pd.concat([cust_attr, ex_director],axis=1)
        cust_attr = pd.concat([cust_attr, cust_payment],axis=1)
        #cust_attr = self.cont_rec.group_size(sales_file, index_col='顧客ID', keep_list=['顧客ID','指名回数','コース受託回数','紹介カード受渡回数','治療送客回数','院長挨拶回数'])

        # 集計処理2.2：顧客IDごとの個別商品属性情報を集計
        ex_id_product = file['顧客ID']
        ex_product_code = file['商品コード']
        ex_price_product = file['売上単価']
        ex_amount_product = file['数量']
        # マージ
        product_attr = pd.concat([ex_id_product, ex_product_code],axis=1)
        product_attr = pd.concat([product_attr, ex_price_product],axis=1)
        product_attr = pd.concat([product_attr, ex_amount_product],axis=1)
        # 売上列追加
        product_attr['売上'] = file['売上単価'] * file['数量']
        # 個別商品IDに相当する列追加
        product_attr['明細ID'] = file['伝票コード'] * 10 + file['明細コード']
        # 不要な行を削除
        #target_attr = product_attr[(product_attr['商品コード']=='1A1501')|(product_attr['商品コード']=='1B2201')|(product_attr['商品コード']=='1A1601')|(product_attr['商品コード']=='200071')|(product_attr['商品コード']=='200006')]


        # 書き出し処理
        self.file_io.export_csv_from_pandas(cust_payment, self.payment_path)
        self.file_io.export_csv_from_pandas(cust_attr, self.cust_attr_path)
        #self.file_io.export_csv_from_pandas(target_attr, self.target_attr_path)
        self.file_io.export_csv_from_pandas(product_attr, self.average_attr_path)


class ExtractSalesSp:
    def __init__(self, in_path, in_char, payment_path, out_char, cust_attr_path, target_attr_path, average_attr_path):
        self.count_rec = CountRecord()
        self.file_io = FileIO()
        self.in_path = in_path
        self.in_char = in_char
        self.payment_path = payment_path
        self.out_char = out_char
        self.cust_attr_path = cust_attr_path
        self.target_attr_path = target_attr_path
        self.average_attr_path = average_attr_path

    def extract(self):
        # ファイルオープン処理
        file = self.file_io.open_file_as_pandas(self.in_path,self.in_char)

        # 顧客属性前処理：顧客属性取得のため、個別商品の行を売上の行に統合
        sales_file = file.query('明細コード == 1')

        # 集計処理：顧客IDごとの支払情報を集計
        cust_payment = self.count_rec.group_sum(sales_file, index_col='顧客ID', aggregate_col=['顧客ID','施術時間'])

        # 顧客属性集計処理：顧客IDごとの属性情報を集計
        ex_id = sales_file['顧客ID']
        ex_nominate = sales_file['指名回数']
        ex_course = sales_file['コース受諾回数']
        ex_card = sales_file['紹介カード受渡回数']
        ex_reception = sales_file['治療送客回数']
        ex_director = sales_file['院長挨拶回数']
        # 追加顧客属性
        #ex_branch = sales_file['店舗']
        #ex_accosiate = sales_file['担当者']

        # マージ
        cust_attr = pd.concat([ex_id, ex_nominate],axis=1)
        cust_attr = pd.concat([cust_attr, ex_course],axis=1)
        cust_attr = pd.concat([cust_attr, ex_card],axis=1)
        cust_attr = pd.concat([cust_attr, ex_reception],axis=1)
        cust_attr = pd.concat([cust_attr, ex_director],axis=1)
        cust_attr = pd.concat([cust_attr, cust_payment],axis=1)
        #cust_attr = self.cont_rec.group_size(sales_file, index_col='顧客ID', keep_list=['顧客ID','指名回数','コース受託回数','紹介カード受渡回数','治療送客回数','院長挨拶回数'])

        # 集計処理2.2：顧客IDごとの個別商品属性情報を集計
        ex_id_product = file['顧客ID']
        ex_product_code = file['商品コード']
        ex_price_product = file['売上単価']
        ex_amount_product = file['数量']
        # マージ
        product_attr = pd.concat([ex_id_product, ex_product_code],axis=1)
        product_attr = pd.concat([product_attr, ex_price_product],axis=1)
        product_attr = pd.concat([product_attr, ex_amount_product],axis=1)
        # 売上列追加
        product_attr['売上'] = file['売上単価'] * file['数量']
        # 個別商品IDに相当する列追加
        product_attr['明細ID'] = file['伝票コード'] * 10 + file['明細コード']
        # スコア列設定
        product_attr['スコア'] = 0
        # スコア設定
        product_attr.loc[product_attr['商品コード']=='1A1501','スコア'] = 5
        product_attr.loc[product_attr['商品コード']=='1B2201','スコア'] = 4
        product_attr.loc[product_attr['商品コード']=='1A1601','スコア'] = 3
        product_attr.loc[product_attr['商品コード']=='200071','スコア'] = 2
        product_attr.loc[product_attr['商品コード']=='200006','スコア'] = 1
        product_attr['スコア'] = product_attr['スコア'] * product_attr['数量']
        # 不要な行を削除
        #product_attr = product_attr[(product_attr['商品コード']=='1A1501')|(product_attr['商品コード']=='1B2201')|(product_attr['商品コード']=='1A1601')|(product_attr['商品コード']=='200071')|(product_attr['商品コード']=='200006')]


        # 書き出し処理
        self.file_io.export_csv_from_pandas(cust_payment, self.payment_path)
        self.file_io.export_csv_from_pandas(cust_attr, self.cust_attr_path)
        #self.file_io.export_csv_from_pandas(target_attr, self.target_attr_path)
        self.file_io.export_csv_from_pandas(product_attr, self.average_attr_path)
