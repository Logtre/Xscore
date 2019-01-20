print(__doc__)


# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


class Classifier:

    def __init__(self):
        self.file_io = FileIO()
        self.test = Test()
        self.ss = Scaler()
        self.drop_na = DropNaN()

        self.h = .02  # step size in the mesh

        self.names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
                 "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
                 "Naive Bayes", "QDA"]

        self.classifiers = [
            KNeighborsClassifier(3),
            SVC(kernel="linear", C=0.025),
            SVC(gamma=2, C=1),
            GaussianProcessClassifier(1.0 * RBF(1.0)),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            MLPClassifier(alpha=1),
            AdaBoostClassifier(),
            GaussianNB(),
            QuadraticDiscriminantAnalysis()]

    def classify(self, in_path, out_path):

        org_df = self.file_io.open_file_as_pandas(in_path,"utf-8")
        feat_shop = self.file_io.open_file_as_pandas('./data/out/feat_shop.csv','utf-8')
        feat_pref = self.file_io.open_file_as_pandas('./data/out/feat_pref.csv','utf-8')

        # shop追加
        #org_df = pd.merge(org_df, feat_shop, on='顧客ID',how='left')
        org_df = pd.merge(org_df, feat_pref, on='顧客ID',how='left')
        org_df = org_df.drop(['Unnamed: 0_x','Unnamed: 0_y'],axis=1)
        org_df = org_df[org_df.columns.drop(list(org_df.filter(regex='Unnamed:')))]
        # 売上<=0を削除
        org_df = org_df.drop(org_df[org_df['売上']<=0].index)
        # 不要列削除
        #org_df = org_df.drop(['Unnamed: 0', '顧客ID'], axis=1)
        org_df = org_df.drop(['顧客ID'],axis=1)
        # 欠損値をゼロうめ
        org_df = org_df.fillna(0)

        # 目的変数Xと説明変数Y
        Y = org_df['売上']
        #Y = org_df['スコア']
        #X = org_df.drop(['支払合計'],axis=1)
        X = org_df.drop(['売上単価','数量','売上'],axis=1)
        #X = org_df.drop(['商品コード','売上単価','数量','売上','明細ID','スコア'],axis=1)
        X = X.drop(['キャンセル回数','コンタクト回数','問い合わせ回数'],axis=1)
        #X = X.drop(['治療送客回数_あり','治療送客回数_なし','院長挨拶回数_あり','院長挨拶回数_なし','紹介カード受渡回数_あり','紹介カード受渡回数_なし','携帯TEL_有','携帯メール_有','性別_女','性別_男','自宅TEL_有','PCメール_有'],axis=1)
        #X = X.drop(['職業_学生','職業_会社員','職業_主婦','職業_自営業','職業_その他','職業_パート・アルバイト'],axis=1)
        X = X.drop(['登録区分_HP','登録区分_店舗','登録区分_CC'],axis=1)
        X = X.drop(['生年月日','滞在時間','閲覧ページ総数','閲覧ページ数/セッション'],axis=1)
        X = X.drop(['治療送客回数_空欄','指名回数_空欄','コース受諾回数_空欄','紹介カード受渡回数_空欄','院長挨拶回数_空欄','性別_空欄','携帯TEL_空欄','自宅TEL_空欄','携帯メール_空欄','PCメール_空欄','職業_空欄','登録区分_空欄'],axis=1)
        X = X[X.columns.drop(list(org_df.filter(regex='_nan')))]
        X = X[X.columns.drop(list(org_df.filter(regex='_なし')))]
        #X = X[X.columns.drop(list(org_df.filter(regex='_空欄')))]
        X = X[X.columns.drop(list(org_df.filter(regex='_無')))]
        X = X[X.columns.drop(list(org_df.filter(regex='_削除')))]
        X = X[X.columns.drop(list(org_df.filter(regex='施術時間')))]
        X = X[X.columns.drop(list(org_df.filter(regex='性別_男')))]
        X = X[X.columns.drop(list(org_df.filter(regex='性別_女')))]
        X = X[X.columns.drop(list(org_df.filter(regex='携帯TEL_有')))]
        X = X[X.columns.drop(list(org_df.filter(regex='治療送客回数_あり')))]
        X = X[X.columns.drop(list(org_df.filter(regex='紹介カード受渡回数_あり')))]
        X = X[X.columns.drop(list(org_df.filter(regex='町域_')))] # 結果にほとんど関係ないので削除

        # 標準化
        std_X = self.ss.standard_scaler(X,axis=1,data_type='float')
        #std_Y = pd.DataFrame(self.sc.fit_transform(Y))
        #std_Y.columns = Y.columns
        #std_X = pd.DataFrame(self.sc.fit_transform(X))
        #std_X.columns = X.columns

        # 正規化
        #norm_Y = pd.DataFrame(self.ms.fit_transform(Y))
        #norm_Y.columns = Y.columns
        #norm_X = pd.DataFrame(self.ms.fit_transform(X))
        #norm_X.columns = X.columns
        #self.file_io.export_csv_from_pandas(X, './data/out/X.csv')

        # トレーニングデータとテストデータに分割(30%)
        X_train, X_test, Y_train, Y_test = self.test.make_train_test_data(std_X, Y, 0.3)
        #X_train, X_test, Y_train, Y_test = self.test.make_train_test_data(X, Y, 0.3)
        print(X_train.head())
        print("--- X_train's shape ---\n {}\n".format(X_train.shape))
        print(X_test.head())
        print("--- X_test's shape ---\n {}\n".format(X_test.shape))
        print(Y_train.head())
        print("--- Y_train's shape ---\n {}\n".format(Y_train.shape))
        print(Y_test.head())
        print("--- Y_test's shape ---\n {}\n".format(Y_test.shape))

#X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
#                           random_state=1, n_clusters_per_class=1)


#rng = np.random.RandomState(2)
#X += 2 * rng.uniform(size=X.shape)
#linearly_separable = (X, y)

#datasets = [make_moons(noise=0.3, random_state=0),
#            make_circles(noise=0.2, factor=0.5, random_state=1),
#            linearly_separable
#            ]

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   edgecolors='k', alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

plt.tight_layout()
plt.show()
