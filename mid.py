'''
列出所有更改的內容：
1. 缺漏值填補：從 IterativeImputer 改為 SimpleImputer 使用單變量插補。
2. 名目屬性轉換：從 Target Encoding 改為 One-Hot Encoding。
3. 離群值處理：從 IQR 方法改為 Z-score 方法。
4. 資料數量化編（減量抽樣）：從 ClusterCentroids 改為 Random Under-Sampling。
5. 交叉驗證方式：從 K-Fold Cross Validation 改為 Leave-One-Out Cross Validation。
'''

import copy
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import LeaveOneOut
from sklearn import svm
from scipy.stats import zscore

from imblearn.under_sampling import RandomUnderSampler



def _print_class_counts(df_y):
    for series_name, series in df_y.value_counts().items():
        print("Y類別值({0})：{1}".format(series_name, series))

def load_data() -> (pd.DataFrame, pd.DataFrame):
    print("載入乳癌資料")
    iris = load_breast_cancer(as_frame=True)
    df_X, df_y = iris.data, iris.target
    print("樣本數量：{0}".format(df_X.shape[0]))
    print("屬性個數：".format(df_X.shape[1]))
    print("樣本數/屬性數：{0}".format(round(df_X.shape[0] / df_X.shape[1], 3)))
    print("==================")
    _print_class_counts(df_y)
    return df_X, df_y

def value_impute(df: pd.DataFrame, df1: pd.DataFrame) -> np.array:
    """ [parameters]
    df: 參考補值的資料
    df1: 欲進行補值的資料
    """
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    # 設定缺漏值以np的nan型式呈現，補值策略為平均值(mean)
    imp.fit(df)
    return pd.DataFrame(imp.transform(df1), columns=df1.columns)

def one_hot_encoding(df_X: pd.DataFrame) -> pd.DataFrame:
    """ 使用One-Hot Encoding方式進行名目屬性轉換
    [parameters]
    df_X: 輸入項X
    """
    # 先檢查dataframe是否包含名目屬性
    list_nominal_attribute_names = []
    for series_name, series in df_X.items():
        if series.dtype.name in ['category', 'object']:
            list_nominal_attribute_names.append(series_name)
    # 如果包含名目屬性
    if len(list_nominal_attribute_names) > 0:
        oh_encoder = preprocessing.OneHotEncoder(handle_unknown='ignore', sparse=False)
        encoded_features = oh_encoder.fit_transform(df_X[list_nominal_attribute_names])
        encoded_df = pd.DataFrame(encoded_features, columns=oh_encoder.get_feature_names_out(list_nominal_attribute_names))
        df_X = df_X.drop(list_nominal_attribute_names, axis=1)
        df_X = pd.concat([df_X.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    return df_X

def remove_duplicates(df_X: pd.DataFrame, df_y: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    移除重複資料
    [parameters]
    df_X: 輸入項X
    df_y: 輸出項y
    """
    # 先將輸入資料(X)與輸出資料(y)合併，因重複資料指「X與y均相同」
    df = copy.deepcopy(df_X)
    df[df_y.name] = df_y
    df = df.drop_duplicates() # 將合併的資料中，重複者移除
    return df.iloc[:, :-1], df.iloc[:, -1] # df.shape[0]為資料筆數；df.shape[1]為屬性個數

def remove_outliers(df_X: pd.DataFrame, df_y: pd.DataFrame, times: float = 3) -> (pd.DataFrame, pd.DataFrame):
    """
    移除離群值
    [parameters]
    df_X: 輸入項X
    df_y: 輸出項y
    times: 標準差倍率
    """
    # 先將輸入資料(X)與輸出資料(y)合併，需要移除時一併刪除對應的y
    df = copy.deepcopy(df_X)
    df[df_y.name] = df_y
    y_name = df_y.name
    for column_name in df_X.columns:
        z_scores = zscore(df[column_name])
        filter = (z_scores < times) & (z_scores > -times)
        df = df.loc[filter]
    return df.iloc[:, :-1], df.iloc[:, -1] # df.shape[0]為資料筆數；df.shape[1]為屬性個數

def under_sampling(df_X: pd.DataFrame, df_y: pd.DataFrame, random_state = 0) -> (pd.DataFrame, pd.DataFrame):
    """
    減量抽樣(使得資料中，y的各類別值的數量約略相等)
    [parameters]
    df_X: 輸入項X
    df_y: 輸出項y
    random_state: 隨機狀態
    """
    rus = RandomUnderSampler(random_state=random_state)
    np_array_X, np_array_y = rus.fit_resample(df_X.values, df_y.values)
    return pd.DataFrame(np_array_X, columns=df_X.columns), pd.Series(data=np_array_y, name=df_y.name)

if __name__ == "__main__":
    
    def __print_data_class_info(df_y: pd.DataFrame, msg: str = None):
        if msg is not None:
            print(msg)
        _print_class_counts(df_y)

    repeat_times = 10    

    # 載入資料
    df_X, df_y = load_data()
    # 缺漏值補充
    df_X = value_impute(df_X, df_X)
    # 名目屬性轉換
    df_X = one_hot_encoding(df_X)
    
    # LeaveOneOut檢驗法
    loo = LeaveOneOut()
    loo.get_n_splits(df_X)
    list_mapes = []
    for i, (train_index, test_index) in enumerate(loo.split(df_X)):
        X_train, X_test, y_train, y_test = df_X.iloc[train_index], df_X.iloc[test_index], df_y.iloc[train_index], df_y.iloc[test_index]
        # 移除重複資料
        X_train, y_train = remove_duplicates(X_train, y_train)
        __print_data_class_info(y_train, "==================\n移除重複資料")
        # 移除包含離群值的資料
        X_train, y_train = remove_outliers(X_train, y_train)
        __print_data_class_info(y_train, "==================\n移除包含離群值的資料")

        __print_data_class_info(y_train, "==================\n訓練資料集經過數量化編前")
        # 資料數量化編
        X_train, y_train = under_sampling(X_train, y_train)
        __print_data_class_info(y_train, "==================\n訓練資料集經過數量化編後")

        # 資料標準化
        scaler = preprocessing.MinMaxScaler(feature_range = (0, 1)) #轉換到[1,2]間，如果沒有設定則為[0,1]
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)

        clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
        accuracy = clf.score(scaler.transform(X_test), y_test) # 要記得把X_test也一併使用同一個scaler進行轉換
        print("第{0}次測試資料集的預測準確率：{1}%".format(i + 1, round(accuracy * 100, 3)))
        list_mapes.append(accuracy)

    print("\n============\n平均正確率\n============\n{0}%".format(round((sum(list_mapes) / len(list_mapes)) * 100, 3)))
