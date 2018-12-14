# -*- coding:utf-8 -*-

# 設定ファイル読み込み用
import configparser

from extract_log import ExtractLog

inifile = configparser.ConfigParser()
inifile.read('./config.ini', 'UTF-8')

# --- log ---
ex_log = ExtractLog(
                inifile.get('log', 'csv_in_path'),
                inifile.get('log', 'csv_in_char'),
                inifile.get('log', 'stay_time_path'),
                inifile.get('general', 'char_type'),
                inifile.get('log', 'pv_sum_path'),
                inifile.get('log', 'session_path'))

ex_log.extract()
