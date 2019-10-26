import numpy as np
from sklearn.datasets import load_iris

# 载入iris数据集
data = load_iris()
# 数据分布
X = data['data']
y = data['target']

# ## 代码开始
n, d = X.shape # 样本个数 特征个数
means = np.mean(X,axis=0,keepdims=True) # 数据X的各特征均值
stds = np.std(X,axis=0) # 数据X的个特征标准差
# ## 代码结束
print('样本个数为：'+str(n) + '\n特征个数为：'+str(d))
print('样本各特征均值为：\n')
print(means)
print('样本各特征方差为：\n')
print(stds**2)
# 根据方差特征选择
from sklearn.feature_selection import VarianceThreshold

# ## 代码开始
sel = VarianceThreshold(threshold=0.6) # 选择0.6作为方差阈值
X_new = sel.fit_transform(X) # 经过选择后的特征
# ## 代码结束

stds_new = np.std(X_new, axis=0)
print('选择特征后样本各特征方差为：\n')
print(stds_new**2)

print(X[0:3])
print(X_new[0:3])
