import copy
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn import svm

from imblearn.under_sampling import ClusterCentroids

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

def value_impute(df: pd.DataFrame, df1: pd.DataFrame, random_state = 0) -> np.array:
    """ 使用多變量方式進行缺漏值填補 
    [parameters]
    df: 參考補值的資料
    df1: 欲進行補值的資料
    random_state: 隨機狀態
    """
    imp = IterativeImputer(max_iter=10, random_state = random_state)
    imp.fit(df)
    return pd.DataFrame(imp.transform(df1),columns = df1.columns)

def target_encoding(df_X: pd.DataFrame, df_y: pd.DataFrame) -> pd.DataFrame:
    """ 使用target_encoding方式進行名目屬性轉換
    [parameters]
    df_X: 輸入項X
    df_y: 輸出項y
    """
    # 先檢查dataframe是否包含名目屬性
    list_nominal_attribute_names = []
    for series_name, series in df_X.items():
        if series.dtype.name in ['category', 'object']:
            list_nominal_attribute_names.append(series_name)
    # 如果包含名目屬性
    if len(list_nominal_attribute_names) > 0:
        t_encoder = preprocessing.TargetEncoder(smooth="auto")
        np_array = t_encoder.fit_transform(df_X[list_nominal_attribute_names], df_y)
        d = { list_nominal_attribute_names[attribute_index]: np_array[: attribute_index] for attribute_index in range(len(list_nominal_attribute_names)) }
        df_X.drop(list_nominal_attribute_names, axis=1)
        return pd.merge(pd.DataFrame(d), df_X.drop(list_nominal_attribute_names, axis=1), left_index=True, right_index=True)
    else:
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

def remove_outliers(df_X: pd.DataFrame, df_y: pd.DataFrame, times: float = 1.5) -> (pd.DataFrame, pd.DataFrame):
    """
    移除離群值
    [parameters]
    df_X: 輸入項X
    df_y: 輸出項y
    times: IQR倍率
    """
    # 先將輸入資料(X)與輸出資料(y)合併，需要移除時一併刪除對應的y
    df = copy.deepcopy(df_X)
    df[df_y.name] = df_y
    y_name = df_y.name
    list_dataframes = []
    for class_name, series in df_y.value_counts().items(): # 需要依據y的類別值，將資料分割後，分別進行離群值移除
        df_temp = df[df[y_name] == class_name]
        print(df_temp.shape)
        for column_name, series in df_temp.items():
            if column_name == y_name:
                continue
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1 #IQR is interquartile range.
            ub = Q3 + times * IQR
            lb = Q1 - times * IQR
            filter = (df_temp[column_name] >= lb) & (df_temp[column_name] <= ub)
            df_temp = df_temp.loc[filter]
        list_dataframes.append(df_temp.loc[filter])
            
    df = pd.concat(list_dataframes, axis=0, ignore_index=True)
    return df.iloc[:, :-1], df.iloc[:, -1] # df.shape[0]為資料筆數；df.shape[1]為屬性個數

def under_sampling(df_X: pd.DataFrame, df_y: pd.DataFrame, random_state = 0) -> (pd.DataFrame, pd.DataFrame):
    """
    減量抽樣(使得資料中，y的各類別值的數量約略相等)
    [parameters]
    df_X: 輸入項X
    df_y: 輸出項y
    random_state: 隨機狀態
    """
    list_X_attribute_names = [series_name for series_name, _ in df_X.items()]
    cc = ClusterCentroids(random_state = random_state)
    np_array_X, np_array_y = cc.fit_resample(df_X.values, df_y.values)
    # np_array_X 為 列、欄排列，需要轉置為 欄、列排列
    np_array_X = np_array_X.T
    dict_X = { list_X_attribute_names[attribute_index] : np_array_X[attribute_index] for attribute_index in range(len(list_X_attribute_names)) }
    return pd.DataFrame(dict_X), pd.Series(data = np_array_y, name = df_y.name)


if __name__ == "__main__":
    
    def __print_data_class_info(df_y: pd.DataFrame, msg: str = None):
        if msg is not None:
            print(msg)
        _print_class_counts(df_y)

    repeat_times = 10    

    # 載入資料
    df_X, df_y = load_data()
    # 缺漏值填補
    df_X = value_impute(df_X, df_X)
    # 名目屬性轉換
    df_X = target_encoding(df_X, df_y)
    
    # 重複執行repeat_times次的分層交互驗證
    list_mapes = []
    for times in range(repeat_times):
        # 分層交互驗證
        skf = StratifiedKFold(n_splits = 10, random_state= times, shuffle = True) # n_splits即K值
        skf.get_n_splits(df_X, df_y)
        for i, (train_index, test_index) in enumerate(skf.split(df_X, df_y)):
        
            X_train, X_test, y_train, y_test = df_X.iloc[train_index], df_X.iloc[test_index], df_y.iloc[train_index], df_y.iloc[test_index]
            # 移除重複資料
            X_train, y_train = remove_duplicates(X_train, y_train)
            __print_data_class_info(y_train, "==================\n移除重複資料")
            # 移除包含離群值的資料
            X_train, y_train = remove_outliers(X_train, y_train)
            __print_data_class_info(y_train, "==================\n移除包含離群值的資料")

            __print_data_class_info(y_train, "==================\n訓練資料集經過數量化約前")
            # 資料數量化約
            X_train, y_train = under_sampling(X_train, y_train)
            __print_data_class_info(y_train, "==================\n訓練資料集經過數量化約後")

            # 資料標準化
            scaler = preprocessing.MinMaxScaler(feature_range = (0, 1)) #轉換到[1,2]間，如果沒有設定則為[0,1]
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)

            clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
            accuracy = clf.score(scaler.transform(X_test), y_test) # 要記得把X_test也一併使用同一個scaler進行轉換
            print("第{0}輪第{1}次測試資料集的預測準確率：{2}%".format(times + 1, i + 1, round(accuracy * 100, 3)))
            list_mapes.append(accuracy)

    print("\n============\n平均正確率\n============\n{0}%".format(round((sum(list_mapes) / len(list_mapes)) * 100, 3))) #96.523%
