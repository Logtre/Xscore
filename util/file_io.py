# -*- coding:utf-8 -*-
import pandas as pd

class FileIO:

    def __init__(self):
        pass

    def open_file(self, file_path, mode, encoding):
        file = open(file_path,mode=mode,encoding=encoding)
        return file

    def open_file_as_pandas(self, file_path, encoding):
        file = pd.read_csv(file_path,encoding=encoding)
        return file

    def export_csv_from_pandas(self, df, file_path):
        df.to_csv(file_path)
        return

    def add_csv_from_pandas(self, df, file_path):
        df.to_csv(file_path, mode='a')
        return
        
    def close_file(self, file_obj):
        file_obj.close
        return
