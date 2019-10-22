import numpy as np
from sklearn.datasets import load_iris

# step1：收集数据
data = load_iris()
X = data['data']
y = data['target']

# step2：数据准备
# 训练集划分
train_X1 = X[0:40]
train_y1 = y[0:40]
train_X2 = X[50:90]
train_y2 = y[50:90]
train_X3 = X[100:140]
train_y3 = y[100:140]
train_X = np.r_[train_X1, train_X2, train_X3]
train_y = np.r_[train_y1, train_y2, train_y3]

# 验证计划分
val_X1 = X[40:50]
val_y1 = y[40:50]
val_X2 = X[90:100]
val_y2 = y[90:100]
val_X3 = X[140:150]
val_y3 = y[140:150]
val_X = np.r_[val_X1, val_X2, val_X3]
val_y = np.r_[val_y1, val_y2, val_y3]
# 数据预处理（归一化）
from sklearn import preprocessing

# ## 代码开始
scaler = preprocessing.StandardScaler().fit(train_X) # 算方差
train_X = scaler.transform(train_X) # 将训练数据归一化
val_X = scaler.transform(val_X) # 将验证数据归一化
# ## 代码结束

# step3：选择一个模型
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(10)

# step4：训练
knn.fit(train_X, train_y)

# step5：评估
acc = knn.score(val_X, val_y)
print('验证集准确率：'+str(acc))
acc = knn.score(train_X, train_y)
print('训练集准确率：'+str(acc))