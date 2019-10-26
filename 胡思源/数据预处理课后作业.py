# 导入所需工具包
import numpy as np
# 载入工具包
from sklearn.datasets import load_boston

#训练数据预处理
# 载入原始数据
dataset = load_boston()
x_data = dataset.data # 导入所有特征变量
y_data = dataset.target # 导入目标值（房价）
name_data = dataset.feature_names #导入特征名

### 代码开始
# 填写存在None行的代码
n,d = x_data.shape # 样本个数，特征个数
print('样本个数:' + str(n) + ' 特征个数' + str(d))
train_data = x_data[0:10,[1,5,10]] # 抽取样本和特征，选择第0-9个样本,第1,5,10个特征
print('\n抽取后的样本:')
print(train_data)
### 代码结束

# 生成缺失数据
train_data_missing = train_data
train_data_missing[[0,2,5,7,9],[1,2,0,0,2]] = np.nan#第0行第1列赋值为NaN
print('\n人为制造缺失数据后的样本:')
print(train_data_missing)

# 使用均值补全丢失数据
from sklearn.preprocessing import Imputer

### 代码开始
# 填写存在None行的代码
imp = Imputer(missing_values='NaN',strategy='mean', axis=0)#计算每一列数据的均值
imp.fit(train_data_missing)#fit原始缺失数据
train_data_impute = imp.transform(train_data_missing)
### 代码结束

print('补全前的数据：')
print(train_data_missing)
print('\n补全后的数据：')
print(train_data_impute)

# 对补全后的数据进行均值中心化，方差规模化
from sklearn import preprocessing

### 代码开始
# 填写存在None行的代码
scaler = preprocessing.StandardScaler().fit(train_data_impute)
train_data_scaler = scaler.transform(train_data_impute)
### 代码结束

print('中心化规模化前的数据：')
print(train_data_impute)
print('\n中心化规模化前的数据均值：')
print(np.mean(train_data_impute, axis=0))
print('\n中心化规模化前的数据方差：')
print(np.std(train_data_impute, axis=0))

print('\n中心化规模化后的数据：')
print(train_data_scaler)
print('\n中心化规模化后的数据均值：')
print(np.mean(train_data_scaler, axis=0))
print('\n中心化规模化后的数据方差：')
print(np.std(train_data_scaler, axis=0))


# 对补全后的数据特征进行缩放[0,1]

### 代码开始
# 填写存在None行的代码

min_max_scaler = preprocessing.MinMaxScaler()#将模型从包里取出来，min_max_scaler就是模型
train_data_minmax = min_max_scaler.fit_transform(train_data_impute)
### 代码结束

print('[0,1]缩放前的数据：')
print(train_data_impute)

print('\n[0,1]缩放后的数据：')
print(train_data_minmax)

# 对补全后的训练数据进行特征缩放[-1,+1]

### 代码开始
# 填写存在None行的代码
max_abs_scaler = preprocessing.MaxAbsScaler() #max_abs_scaler模型
train_data_maxabs = max_abs_scaler.fit_transform(train_data_impute)#用模型来训练
### 代码结束

print('[-1,+1]缩放前的数据：')
print(train_data_impute)

print('\n[-1,+1]缩放后的数据：')
print(train_data_maxabs)

# 对补全后的训练数据进行二值化(大于6为1，小于6为0)

### 代码开始
# 填写存在None行的代码
binarizer = preprocessing.Binarizer(threshold=6)
train_data_binarizer = binarizer.transform(train_data_impute)
### 代码结束

print('二值化前的数据：')
print(train_data_impute)

print('二值化]后的数据：')
print(train_data_binarizer)



#测试数据预处理
# 抽取测试数据

### 代码开始
# 填写存在None行的代码
test_data = x_data[10:15,[1,5,10]] # 抽取样本和特征，选择第10-15个样本,第1,5,10个特征
test_data_missing = test_data.copy()
test_data_missing[[0,2,3],[1,2,0]] = np.nan
### 代码结束

print('抽取的测试样本：')
print(test_data)
print('\n人为制造缺失数据后的测试样本')
print(test_data_missing)

# 使用训练数据的均值对缺失值进行补全

### 代码开始
# 填写存在None行的代码
test_data_impute = imp.transform(test_data_missing)
### 代码结束

print('补全前的数据：')
print(test_data_missing)
print('\n补全后的数据：')
print(test_data_impute)


# 根据训练数据对测试数据中心化，方差规模化

### 代码开始
# 填写存在None行的代码
test_data_scaler = scaler.transform(test_data_impute)
### 代码结束

print('中心化规模化前的数据：')
print(test_data_impute)
print('\n中心化规模化前的数据均值：')
print(np.mean(test_data_impute, axis=0))
print('\n中心化规模化前的数据方差：')
print(np.std(test_data_impute, axis=0))

print('\n中心化规模化后的数据：')
print(test_data_scaler)
print('\n中心化规模化后的数据均值：')
print(np.mean(test_data_scaler, axis=0))
print('\n中心化规模化后的数据方差：')
print(np.std(test_data_scaler, axis=0))


# 根据训练数据对补全后的测试数据缩放[0,1]

### 代码开始
# 填写存在None行的代码
test_data_minmax = min_max_scaler.transform(test_data_impute)
### 代码结束

print('[0,1]缩放前的数据：')
print(test_data_impute)

print('\n[0,1]缩放后的数据：')
print(test_data_minmax)

# 根据训练数据对测试数据缩放[-1,1]

### 代码开始
# 填写存在None行的代码
test_data_maxabs = max_abs_scaler.transform(test_data_impute)
### 代码结束

print('[-1,+1]缩放前的数据：')
print(test_data_impute)

print('\n[-1,+1]缩放后的数据：')
print(test_data_maxabs)

# 根据训练数据对补全后的测试数据二值化
test_data_binarizer =binarizer.transform(test_data_impute)

print('二值化前的数据：')
print(test_data_impute)

print('\n二值化]后的数据：')
print(test_data_binarizer)

