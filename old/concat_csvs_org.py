# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from util import CategoryEncode
from util import ExtractColumns
from file_io import FileIO
from util import CountRecord

import datetime
import math


class ConcatCsvs:
    def __init__(
        self,
        id_path,
        cust_payment_path,
        cust_attr_path,
        product_attr_path,
        cust_path,
        cancel_path,
        contact_path,
        cti_path,
        register_type_path,
        status_path,
        stay_time_path,
        pv_sum_path,
        session_path,
        char_type):

        self.file_io = FileIO()
        self.encode = CategoryEncode()
        self.count_rec = CountRecord()
        self.extract_col = ExtractColumns()
        # ファイルオープン
        self.id = self.file_io.open_file_as_pandas(id_path,char_type)
        self.cust_payment = self.file_io.open_file_as_pandas(cust_payment_path, char_type)
        self.cust_attr = self.file_io.open_file_as_pandas(cust_attr_path, char_type)
        self.product_attr = self.file_io.open_file_as_pandas(product_attr_path, char_type)
        self.cust = self.file_io.open_file_as_pandas(cust_path, char_type)
        self.cancel = self.file_io.open_file_as_pandas(cancel_path, char_type)
        self.contact = self.file_io.open_file_as_pandas(contact_path, char_type)
        self.cti = self.file_io.open_file_as_pandas(cti_path, char_type)
        self.register_type = self.file_io.open_file_as_pandas(register_type_path, char_type)
        self.status = self.file_io.open_file_as_pandas(status_path, char_type)
        self.stay_time = self.file_io.open_file_as_pandas(stay_time_path, char_type)
        self.pv_sum = self.file_io.open_file_as_pandas(pv_sum_path, char_type)
        self.session = self.file_io.open_file_as_pandas(session_path, char_type)


    def concat(self, out_path, out_path2):
        # 特徴量抽出処理

        # cust_payment
        # カテゴリーデータなし
        # --- check ---
        #print("--- cust_payment shape ---\n {}\n".format(self.cust_payment.shape))
        #print(self.cust_payment.head())

        # cust_attr
        cust_attr_col_list =[]
        cust_attr_tg_list = ['指名回数','コース受諾回数','紹介カード受渡回数','治療送客回数','院長挨拶回数']
        # カテゴリ列を抽出
        cust_attr_category_col = self.extract_col.extract(self.cust_attr, self.cust_attr['顧客ID'], extract_col=cust_attr_tg_list)
        # 非カテゴリ列を抽出
        cust_attr_non_category_col = self.extract_col.exclude(self.cust_attr, exclude_col=cust_attr_tg_list)
        # 特徴量抽出
        org_cust_attr = self.encode.transform_feature(cust_attr_category_col, aggregate_col=cust_attr_tg_list)
        org_cust_attr = org_cust_attr.fillna(0)
        #org_cust_attr = org_cust_attr.drop('Unnamed: 0', axis=1)
        # ラベル付与
        for col in cust_attr_tg_list:
            cust_attr_col_list += self.encode.transform_label(self.cust_attr[col],col)
        else:
            cust_attr_col_list += ['顧客ID']
        # ラベル設定
        org_cust_attr.columns = cust_attr_col_list
        # 集計処理
        feat_cust_attr = self.count_rec.group_sum(org_cust_attr, index_col='顧客ID', aggregate_col=cust_attr_col_list)
        # カテゴリ列と非カテゴリ列を結合
        feat_cust_attr = pd.merge(feat_cust_attr, cust_attr_non_category_col, on='顧客ID',how='left')
        feat_cust_attr = feat_cust_attr.drop('Unnamed: 0', axis=1)
        # --- check ---
        #print("--- feat_cust_attr shape ---\n {}\n".format(feat_cust_attr.shape))
        #print(feat_cust_attr.head())
        #self.file_io.export_csv_from_pandas(feat_cust_attr, './data/out/mid_feat_cust_attr.csv')

        # product_attr
        product_attr_col_list = []
        product_attr_tg_list = ['商品コード','売上区分','商品区分']
        # カテゴリ列を抽出
        product_attr_category_col = self.extract_col.extract(self.product_attr, self.product_attr['顧客ID'], extract_col=product_attr_tg_list)
        # 非カテゴリ列を抽出
        product_attr_non_category_col = self.extract_col.exclude(self.product_attr, exclude_col=product_attr_tg_list)
        # 特徴量抽出
        org_product_attr = self.encode.transform_feature(product_attr_category_col, aggregate_col=product_attr_tg_list)
        org_product_attr = org_product_attr.fillna(0)
        #org_product_attr = org_product_attr.drop('Unnamed: 0', axis=1)
        #print(org_product_attr)
        # ラベル付与
        for col in product_attr_tg_list:
            product_attr_col_list += self.encode.transform_label(self.product_attr[col],col)
        else:
            product_attr_col_list += ['顧客ID']
        # ラベル設定
        org_product_attr.columns = product_attr_col_list
        # 集計処理
        feat_product_attr = self.count_rec.group_sum(org_product_attr, index_col='顧客ID', aggregate_col=product_attr_col_list)
        # カテゴリ列と非カテゴリ列を結合
        feat_product_attr = pd.merge(feat_product_attr, product_attr_non_category_col, on='顧客ID',how='left')
        feat_product_attr = feat_product_attr.drop('Unnamed: 0', axis=1)
        # --- check ---
        #print("--- feat_product_attr shape ---\n {}\n".format(feat_cust_attr.shape))
        #print(feat_product_attr.head())
        #self.file_io.export_csv_from_pandas(feat_product_attr, './data/out/mid_feat_product_attr.csv')

        # cust
        cust_col_list = []
        cust_tg_list = ['性別','携帯TEL','自宅TEL','携帯メール','PCメール','職業']
        # 外れ値を削除
        new_cust = self.cust.drop(self.cust[self.cust['生年月日'].str.contains('\*', na=True)].index)
        today = int(pd.to_datetime('today').strftime('%Y%m%d'))
        new_cust['生年月日'] = pd.to_datetime(new_cust['生年月日']).dt.strftime('%Y%m%d').astype(np.int64)
        new_cust['生年月日'] = ((today - new_cust['生年月日'])/10000).astype(np.int64)
        # カテゴリ列を抽出
        cust_category_col = self.extract_col.extract(new_cust, new_cust['顧客ID'], extract_col=cust_tg_list)
        # 非カテゴリ列を抽出
        cust_non_category_col = self.extract_col.exclude(new_cust, exclude_col=cust_tg_list)
        # 特徴量抽出
        feat_cust = self.encode.transform_feature(cust_category_col, aggregate_col=cust_tg_list)
        feat_cust = feat_cust.fillna(0)
        #feat_cust = feat_cust.drop('Unnamed: 0', axis=1)
        feat_cust = feat_cust[feat_cust.columns.drop(list(feat_cust.filter(regex='Unnamed:')))]
        # ラベル付与
        for col in cust_tg_list:
            cust_col_list += self.encode.transform_label(new_cust[col],col)
        else:
            cust_col_list += ['顧客ID']
        # ラベル設定
        feat_cust.columns = cust_col_list
        # カテゴリ列と非カテゴリ列を結合
        feat_cust = pd.merge(feat_cust, cust_non_category_col, on='顧客ID',how='left')
        #feat_cust = feat_cust.drop('Unnamed: 0', axis=1)
        # --- check ---
        #print("--- feat_cust shape ---\n {}\n".format(feat_cust.shape))
        #print(feat_cust.head())
        #self.file_io.export_csv_from_pandas(feat_cust, './data/out/mid_feat_cust.csv')

        # cancel
        # カテゴリーデータなし
        # --- check ---
        #print("--- cancel shape ---\n {}\n".format(cancel.shape))
        #print(cancel.head())

        # contact
        # カテゴリーデータなし
        # --- check ---
        #print("--- contact shape ---\n {}\n".format(contact.shape))
        #print(contact.head())

        # cti
        # カテゴリーデータなし
        # --- check ---
        #print("--- cti shape ---\n {}\n".format(cti.shape))
        #print(cti.head())

        # register_type
        reg_col_list = []
        reg_tg_list = ['登録区分']
        # カテゴリ列を抽出
        reg_category_col = self.extract_col.extract(self.register_type, self.register_type['顧客ID'], extract_col=reg_tg_list)
        # 非カテゴリ列を抽出
        reg_non_category_col = self.extract_col.exclude(self.register_type, exclude_col=reg_tg_list)
        # 特徴量抽出
        feat_register_type = self.encode.transform_feature(reg_category_col, aggregate_col=reg_tg_list)
        feat_register_type = feat_register_type.fillna(0)
        #feat_register_type = feat_register_type.drop('Unnamed: 0', axis=1)
        # ラベル付与
        for col in reg_tg_list:
            reg_col_list += self.encode.transform_label(self.register_type[col],col)
        else:
            reg_col_list += ['顧客ID']
        # ラベル設定
        feat_register_type.columns = reg_col_list
        # カテゴリ列と非カテゴリ列を結合
        feat_register_type = pd.merge(feat_register_type, reg_non_category_col, on='顧客ID',how='left')
        feat_register_type = feat_register_type.drop('Unnamed: 0', axis=1)
        # --- check ---
        #print("--- feat_register_type shape ---\n {}\n".format(feat_register_type.shape))
        #print(feat_register_type.head())
        #self.file_io.export_csv_from_pandas(feat_register_type, './data/out/mid_feat_register_type.csv')

        # status
        stat_col_list = []
        stat_tg_list = ['状況','指名区分']
        # カテゴリ列を抽出
        stat_category_col = self.extract_col.extract(self.status, self.status['顧客ID'], extract_col=stat_tg_list)
        # 非カテゴリ列を抽出
        stat_non_category_col = self.extract_col.exclude(self.status, exclude_col=stat_tg_list)
        # 特徴量抽出
        feat_status = self.encode.transform_feature(stat_category_col, aggregate_col=stat_tg_list)
        feat_status = feat_status.fillna(0)
        #feat_status = feat_status.drop('Unnamed: 0', axis=1)
        # ラベル付与
        for col in stat_tg_list:
            stat_col_list += self.encode.transform_label(self.status[col],col)
        else:
            stat_col_list += ['顧客ID']
        # ラベル設定
        feat_status.columns = stat_col_list
        # カテゴリ列と非カテゴリ列を結合
        feat_status = pd.merge(feat_status, stat_non_category_col, on='顧客ID',how='left')
        feat_status = feat_status.drop('Unnamed: 0', axis=1)
        #feat_status = feat_status.drop('Unnamed: 0', axis=1)
        # --- check ---
        #print("--- feat_status shape ---\n {}\n".format(feat_status.shape))
        #print(feat_status.head())
        #self.file_io.export_csv_from_pandas(feat_status, './data/out/mid_feat_status.csv')

        # 結合処理
        con_file = pd.merge(self.id, self.cust_payment, on='顧客ID', how='left')
        #print("1.1: shape is {}".format(con_file.shape))
        con_file = pd.merge(con_file, self.cancel, on='顧客ID',how='left')
        #print("1.2: shape is {}".format(con_file.shape))
        con_file = pd.merge(con_file, self.contact, on='顧客ID',how='left')
        #print("1.3: shape is {}".format(con_file.shape))
        con_file = pd.merge(con_file, self.cti, on='顧客ID',how='left')
        #print("1.4: shape is {}".format(con_file.shape))
        con_file = pd.merge(con_file, self.stay_time, on='顧客ID',how='left')
        #print("1.5: shape is {}".format(con_file.shape))
        con_file = pd.merge(con_file, self.pv_sum, on='顧客ID',how='left')
        #print("1.6: shape is {}".format(con_file.shape))
        con_file = pd.merge(con_file, self.session, on='顧客ID',how='left')
        #print("1.7: shape is {}".format(con_file.shape))
        con_file = pd.merge(con_file, feat_cust_attr, on='顧客ID',how='left')
        #print("1.8: shape is {}".format(con_file.shape))
        con_file = pd.merge(con_file, feat_cust, on='顧客ID',how='left')
        #print("1.9: shape is {}".format(con_file.shape))
        con_file = pd.merge(con_file, feat_register_type, on='顧客ID',how='left')
        #print("1.10: shape is {}".format(con_file.shape))
        #con_file = pd.merge(con_file, feat_status, on='顧客ID',how='left')
        #print("1.11: shape is {}".format(con_file.shape))
        '''con_file = pd.concat([
            self.cust_payment,
            feat_cust_attr,
            feat_cust,
            self.cancel,
            self.contact,
            self.cti,
            feat_register_type,
            feat_status,
            self.stay_time,
            self.pv_sum,
            self.session], axis=1, join_axes=['顧客ID'])'''
        # --- check ---
        #print("--- con_file shape ---\n {}\n".format(con_file.shape))
        #print(con_file.head())

        # 結合処理
        con_product_file = pd.merge(self.id, self.cust_payment, on='顧客ID', how='left')
        con_product_file = pd.merge(con_product_file, feat_product_attr, on='顧客ID', how='left')

        #print("2.1: shape is {}".format(con_file.shape))

        # 重複がある場合、削除
        con_file = con_file.drop_duplicates()
        con_product_file = con_product_file.drop_duplicates()
        con_product_file = con_product_file.drop(['施術時間','売上単価','数量'],axis=1)

        # 書き出し処理
        self.file_io.export_csv_from_pandas(con_file, out_path)
        self.file_io.export_csv_from_pandas(con_product_file, out_path2)
