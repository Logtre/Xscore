# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
#import matplotlib.font_manager as fm

#fm.findSystemFonts()
plt.rcParams['font.family'] = 'IPAPGothic'

class DrawChart2:
    def __init__(self):
        pass

    def draw(self, lr, X, Y, X_col, Y_col):

        plt.scatter(Y, X) # 散布図をプロット
        #plt.plot(X, lr.predict(X), color = 'red') # 回帰直線をプロット

        plt.title('Scatter Plot of {} vs {}'.format(X_col,Y_col))    # 図のタイトル
        plt.xlabel(X_col) # x軸のラベル
        plt.ylabel(Y_col)    # y軸のラベル
        plt.grid()          # グリッド線を表示

        plt.show()                                 # 図の表示
