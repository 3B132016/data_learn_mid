
列出所有更改的內容：
1. 缺漏值填補：從 IterativeImputer 改為 SimpleImputer，使用單變量插補。
2. 名目屬性轉換：從 Target Encoding 改為 One-Hot Encoding。
3. 離群值處理：從 IQR 方法改為 Z-score 方法。
4. 資料數量化編（減量抽樣）：從 ClusterCentroids 改為 Random Under-Sampling。
5. 交叉驗證方式：從 K-Fold Cross Validation 改為 Leave-One-Out Cross Validation。
