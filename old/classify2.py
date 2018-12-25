# -*- coding: utf-8 -*-
# 設定ファイル読み込み用
import configparser
# 独自ファイル読み込み用
from file_io import FileIO
from util import CountRecord

import numpy as np
import pandas as pd

#%matplotlib inline
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA

file_io = FileIO()
count_rec = CountRecord()

inifile = configparser.ConfigParser()
inifile.read('./config.ini', 'UTF-8')

plt.figure(figsize=(8, 8))
plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)

plt.subplot(321)

datasets = file_io.open_file_as_pandas(inifile.get('classify', 'x_path'),"utf-8")
datasets = datasets.drop(datasets[datasets['売上単価']<=0].index)
#X = datasets.drop(['数量'],axis=1)
#y = datasets['数量']

amount_per_product = count_rec.group_sum(datasets, index_col='商品コード', aggregate_col=['数量'])
price_per_product = datasets.drop_duplicates(subset='商品コード', keep='last')
price_per_product = price_per_product.drop('数量', axis=1)
sales_datasets = pd.merge(amount_per_product, price_per_product, on='商品コード')
sales_datasets['売上'] = sales_datasets['売上単価'] * sales_datasets['数量']
X = sales_datasets['売上']
y = range(232)

cm = plt.cm.get_cmap('RdYlBu')

sc = plt.scatter(y,X,marker='o',c=range(232), vmin=0, vmax=8000, cmap=cm)
#plt.colorbar(sc)
plt.show()
# <matplotlib.colorbar.Colorbar at 0x7f880818e6d0>
