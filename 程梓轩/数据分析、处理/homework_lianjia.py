import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# 版本查看
from sys import version_info
if version_info.major != 3:
    raise Exception('请使用Python 3 来完成此项目')
# 画图预设
plt.style.use("seaborn-pastel")
sns.set_style({'white': ['simhei', 'Arial']})
# 读取文件
lianjia_df = pd.read_csv("D:/big data/python/data/lianjia.csv")
# 基本数据及类型总览
print(lianjia_df.describe())
print()
print(lianjia_df.info())
print()
# 特征少个单价增加上
df = lianjia_df.copy()
df['PerPrice'] = lianjia_df['Price'] / lianjia_df['Size']
columns = ['Region', 'District', 'Garden', 'Layout', 'Floor', 'Year', 'Size', 'Elevator', 'PerPrice', 'Price', 'Id']
df = pd.DataFrame(df, columns=columns)# 把列更新一下
print(df)
# 以year来分析
#   不同Year下的各房屋户型的数量对比
df_yl = df.groupby(['Year', 'Layout'])['Id'].count().to_frame().reset_index()
#       打印统计数目
# print(df_yl)
#       画图
df_yl['YL'] = df_yl['Year'].astype(str) +" "+df_yl['Layout']
df_yl = df_yl.drop(['Year', 'Layout'], axis=1)
# columns2 = ['YL', 'Id']
# df_yl = pd.DataFrame(df_yl, columns=columns2)
print(df_yl)
# y1_label = list(set(df_yl['Year']))
# x_labels = range(len(y1_label))
# df_lt = pd.DataFrame(df_yl['Layout'].value_counts())
# total_width, n = 0.8, df_lt.shape[0]
# width = total_width/n
#   以Year为分类点价格均值
df_y = df.groupby('Year')['Id'].count().to_frame().reset_index()
df_y_price = df.groupby('Year')['Price'].sum().to_frame().reset_index()
means = df_y_price['Price']/df_y['Id']
df_means = pd.concat([df_y['Year'], means], axis=1)
#   以Year为分类点面积均值
df_square = df.groupby('Year')['Size'].sum().to_frame().reset_index()
square_means = df_square['Size']/df_y['Id']
df_square_means = pd.concat([df_y['Year'], square_means], axis=1)
# print(df_square_means)
# 小区房屋特征分析
#   查询房屋平均单价最便宜和最贵的小区名
df_block = df.groupby('Garden')['Price'].mean().to_frame().reset_index().sort_values(by = 'Price', ascending=False)
print("平均单价最贵/最便宜小区名：")
print(df_block.iloc[0, 0]+"/"+df_block.iloc[df_block.shape[0]-1, 0]+"\n")
# 查询房屋平均年份最老和最新的区县名
df_old = df.groupby('Region')['Year'].mean().to_frame().reset_index().sort_values(by = 'Year', ascending=False)
print("最新/最老的区县名：")
print(df_old.iloc[0, 0]+"/"+df_old.iloc[df_old.shape[0]-1, 0]+"\n")
# 09 小区特征聚类分析
# 对电梯值处理
df.loc[(df['Floor'] > 6) & (df['Elevator'].isnull()), 'Elevator'] = '有电梯'
df.loc[(df['Floor'] <= 6) & (df['Elevator'].isnull()), 'Elevator'] = '无电梯'
# 二值化
from sklearn.preprocessing import LabelBinarizer
df['Elevator'] = LabelBinarizer().fit_transform(df['Elevator'])
# 特征数字化
Region_list = list(set(df['Region']))
Region_dic = dict()
i = 0
for j in Region_list:
    Region_dic[j] = i
    i = i+1
District_list = list(set(df['District']))
District_dic = dict()
i = 0
for j in District_list:
    District_dic[j] = i
    i = i+1
Garden_list = list(set(df['Garden']))
Garden_dic = dict()
i = 0
for j in Garden_list:
    Garden_dic[j] = i
    i = i+1
Layout_list = list(set(df['Layout']))
Layout_dic = dict()
i = 0
for j in Layout_list:
    Layout_dic[j] = i
    i = i+1


Region_pro = []
for k in df['Region']:
    Region_pro.append(Region_dic[k])
District_pro = []
for k in df['District']:
    District_pro.append(District_dic[k])
Garden_pro = []
for k in df['Garden']:
    Garden_pro.append(Garden_dic[k])
Layout_pro = []
for k in df['Layout']:
    Layout_pro.append(Layout_dic[k])

df['Region'] = pd.DataFrame(Region_pro, columns=['Region'])
df['District'] = pd.DataFrame(District_pro, columns=['District'])
df['Garden'] = pd.DataFrame(Garden_pro, columns=['Garden'])
df['Layout'] = pd.DataFrame(Layout_pro, columns=['Layout'])
columns = ['Region', 'District', 'Garden', 'Layout', 'Floor', 'Year', 'Size', 'PerPrice', 'Price']
df = pd.DataFrame(df, columns=columns)
# print(df.info())


# print(type(df))
# 聚类
from Kmeans import Kmeans_test
A = Kmeans_test.Kmeans(4, 22, 0, df).demand(100, 0.5)
# A = Kmeans_test.rukou("lianjia",1000,9,22,0,0.5,df)
print(A)

