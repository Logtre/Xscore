# -*- coding:utf-8 -*-

# 設定ファイル読み込み用
import configparser
# 独自ライブラリ
from extract_cancel import ExtractCancel
from extract_contact import ExtractContact
from extract_cti import ExtractCti
from extract_cust import ExtractCust
from extract_reserve import ExtractReserve
from extract_sales import ExtractSales
from extract_log import ExtractLog
from concat_csvs import ConcatCsvs
from linear_regression import LinRegression
from linear_regression2 import LinRegression2
from individual_regression import IndRegression
from select_columns import SelectColumns

inifile = configparser.ConfigParser()
inifile.read('./config.ini', 'UTF-8')


# --- cancel ---
ex_cancel = ExtractCancel(
                inifile.get('cancel', 'csv_in_path'),
                inifile.get('cancel', 'csv_in_char'),
                inifile.get('cancel', 'csv_out_path'),
                inifile.get('general', 'char_type'))
# --- contact ---
ex_contact = ExtractContact(
                inifile.get('contact', 'csv_in_path'),
                inifile.get('contact', 'csv_in_char'),
                inifile.get('contact', 'csv_out_path'),
                inifile.get('general', 'char_type'))
# --- cti ---
ex_cti = ExtractCti(
                inifile.get('cti', 'csv_in_path'),
                inifile.get('cti', 'csv_in_char'),
                inifile.get('cti', 'csv_out_path'),
                inifile.get('general', 'char_type'))
# --- cust ---
ex_cust = ExtractCust(
                inifile.get('cust', 'csv_in_path'),
                inifile.get('cust', 'csv_in_char'),
                inifile.get('cust', 'csv_out_path'),
                inifile.get('general', 'char_type'),
                inifile.get('cust', 'id_path'))
# --- reserve ---
ex_reserve = ExtractReserve(
                inifile.get('reserve', 'csv_in_path'),
                inifile.get('reserve', 'csv_in_char'),
                inifile.get('reserve', 'status_path'),
                inifile.get('general', 'char_type'),
                inifile.get('reserve', 'reg_type_path'))
# --- sales ---
ex_sales = ExtractSales(
                inifile.get('sales', 'csv_in_path'),
                inifile.get('sales', 'csv_in_char'),
                inifile.get('sales', 'payment_path'),
                inifile.get('general', 'char_type'),
                inifile.get('sales', 'cust_attr_path'),
                inifile.get('sales', 'target_attr_path'),
                inifile.get('sales', 'average_attr_path'))
# --- log ---
ex_log = ExtractLog(
                inifile.get('log', 'csv_in_path'),
                inifile.get('log', 'csv_in_char'),
                inifile.get('log', 'stay_time_path'),
                inifile.get('general', 'char_type'),
                inifile.get('log', 'pv_sum_path'),
                inifile.get('log', 'session_path'))
# 処理の実行
# 展開処理
ex_cancel.extract()
ex_contact.extract()
ex_cti.extract()
ex_cust.extract()
ex_reserve.extract()
ex_sales.extract()
ex_log.extract()

# --- concat_csvs ---
concat = ConcatCsvs(
                inifile.get('concat','id_path'),
                inifile.get('sales', 'payment_path'),
                inifile.get('sales', 'cust_attr_path'),
                inifile.get('sales', 'target_attr_path'),
                inifile.get('sales', 'average_attr_path'),
                inifile.get('cust', 'csv_out_path'),
                inifile.get('cancel', 'csv_out_path'),
                inifile.get('contact', 'csv_out_path'),
                inifile.get('cti', 'csv_out_path'),
                inifile.get('reserve', 'reg_type_path'),
                inifile.get('reserve', 'status_path'),
                inifile.get('log', 'stay_time_path'),
                inifile.get('log', 'pv_sum_path'),
                inifile.get('log', 'session_path'),
                inifile.get('general', 'char_type'))

# csvの結合処理
concat.concat(inifile.get('concat', 'csv_out_path'),inifile.get('concat', 'csv_out_path2'))

'''
# --- Individual_regression ---
ir = IndRegression()

# 個別回帰分析
ir.regression(inifile.get('concat', 'csv_out_path'),'./data/out/individual_test_cust.csv')

#ir.regression(inifile.get('concat', 'csv_out_path2'),'./data/out/individual_test_product.csv')
'''

#select = SelectColumns(inifile.get('concat', 'csv_out_path'),inifile.get('general', 'char_type'))
#select.select(extract_col=['売上単価','コース受諾回数_なし','数量','施術時間','指名回数_あり'])

# --- linear_regression ---
lr = LinRegression()

# 重回帰分析
lr.regression(inifile.get('concat', 'csv_out_path'),'./data/out/linear_test_cust.csv')

#lr.regression(inifile.get('concat', 'csv_out_path2'),'./data/out/linear_test_product.csv')

#lr2 = LinRegression2()

# 重回帰分析
#lr2.regression(inifile.get('concat', 'csv_out_path'),'./data/out/linear2_test_cust.csv')
