#########################################
# CODE:   Python数据分析示例代码         #
# AUTHOR: Tiger                         #
# TIME:   2019-09-28                    #
#########################################

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import random

# 设置绘图字体
plt.style.use("fivethirtyeight")
sns.set_style({'font.sans-serif':['simhei','Arial']})

from sys import version_info
if version_info.major != 3:
    raise Exception('请使用Python 3 来完成此项目')

########################################
###########北京二手房租售数据############
########################################
# 读入待分析数据
lianjia_df = pd.read_csv("D:/a/lianjia.csv")
# Region     -- 区县名称
# District   -- 地区
# Garden     -- 小区
# Layout     -- 户型
# Year       -- 房屋建成时间
# Size       -- 房屋面积
# Elevator   -- 是否带电梯
# Direction  -- 房屋朝向
# Renovation -- 房屋装修水平
# PerPrice   -- 房屋单价(新增特征，原数据中没有)
# Price      -- 房屋总价
# Floor      -- 房屋楼层

# 00 基本数值统计特征
# print(lianjia_df.describe())
#    列值类型总览
# print(lianjia_df.info())

# 01 新增列特征、重排数据特征的列顺序
df = lianjia_df.copy()
df['PerPrice'] = lianjia_df['Price'] / lianjia_df['Size']
columns = ['Region', 'District', 'Garden', 'Layout', 'Floor', 'Year', 'Size', 'Elevator', 'Direction', 'Renovation', 'PerPrice', 'Price']
df = pd.DataFrame(df, columns = columns)

# # 02 Region区域特征分析
# 对不同区域的二手房分组对比二手房数量和每平米房价
house_mean = df.groupby(by = 'Region')['PerPrice'].mean().sort_values(ascending=False).to_frame().reset_index()
house_count = df.groupby(by = 'Region')['Size'].count().sort_values(ascending=False).to_frame().reset_index()
# f, [plt1, plt2] = plt.subplots(2, 1, figsize=(20, 10))
# sns.barplot(x = 'Region', y = 'PerPrice', palette = "Blues_d", data = house_mean, ax = plt1)
# plt1.set_title('北京各大区二手房每平米单价对比', fontsize = 12)
# plt1.set_xlabel('')
# sns.barplot(x = 'Region', y = 'Size', palette = "Greens_d", data = house_count, ax = plt2)
# plt2.set_title('北京各大区二手房数量对比',fontsize = 12)
# plt2.set_xlabel('')
# plt.show()

# 03 Size房屋总面积特征分析
# 分析住房面积与住房价格之间的关系，以及住房面积的分布情况
# f, [plt1, plt2] = plt.subplots(1, 2, figsize=(12, 4))
# # 总体房屋面积大小的分布情况
# sns.distplot(df['Size'], bins=50, ax=plt1, kde=True,
#              kde_kws={"color": "r", "lw": 1, 'linestyle': '--'}
#              )
# # 房屋总面积大小和出售价格的关系
# sns.regplot(x='Size', y='Price', data=df, ax=plt2, color = 'b', scatter_kws={"s": 1})
# plt.show()
# 删除面积大小异常值
df = df.loc[(df['Size'] >= 10) & (df['Size'] < 1000)]

# 04 Layout 户型特征分析
# 查看不同户型下的房屋数量
# f, plt1 = plt.subplots(figsize = (20, 20))
# sns.countplot(y = 'Layout', data = df, ax = plt1)
# plt1.set_title('房屋户型', fontsize = 12)
# plt1.set_xlabel('数量', fontsize = 8)
# plt1.set_ylabel('户型', fontsize = 8)
# plt.show()
# 排除特殊户型房屋
df = df[(df['Layout'] != '叠拼别墅') & (df['Size'] < 1000)]


# 05 Renovation 装修特征分析
# 查看不同装修水平的房屋数量(对比04案例中的统计分析方法的不同)
value_distinct = df['Renovation'].value_counts()
# print(value_distinct)
# 去掉错误的装修特征数据“南北”
df['Renovation'] = df.loc[(df['Renovation'] != '南北'), 'Renovation']
# f, [plt1, plt2, plt3] = plt.subplots(1, 3, figsize=(20, 5))
# sns.countplot(df['Renovation'], ax = plt1)
# sns.barplot(x = 'Renovation', y = 'Price', data=df, ax = plt2)
# sns.boxplot(x = 'Renovation', y = 'Price', data=df, ax = plt3)
# plt.show()


# 06 Elevator 房屋电梯特征分析
# 查看Elevator列中有缺失值的行数
misn = len(df.loc[(df['Elevator'].isnull()), 'Elevator'])
# print('Elevator缺失值数量为：'+ str(misn))
# 填补Elevator列中的缺失值（填充准则：假设６楼以上都应该有电梯，６楼以下没有电梯）
df.loc[(df['Floor'] > 6) & (df['Elevator'].isnull()), 'Elevator'] = '有电梯'
df.loc[(df['Floor'] <= 6) & (df['Elevator'].isnull()), 'Elevator'] = '无电梯'
# f, [plt1, plt2] = plt.subplots(1, 2, figsize=(20, 10))
# sns.countplot(df['Elevator'], ax=plt1)
# plt1.set_title('有无电梯数量对比',fontsize=15)
# plt1.set_xlabel('是否有电梯')
# plt1.set_ylabel('数量')
# sns.barplot(x='Elevator', y='Price', data=df, ax=plt2)
# plt2.set_title('有无电梯房价对比',fontsize=15)
# plt2.set_xlabel('是否有电梯')
# plt2.set_ylabel('总价')
# plt.show()

############   以下为本次数据分析的作业   ################

# 07 Year 分析不同建房年份下的房屋特征
# year = df.groupby('Year')
# 不同Year下的各房屋户型的数量对比(可以绘图，也可以用数字呈现；比如：2000年，一室一厅，20；2000年，两室一厅，30）
# print(year.describe())
# for name, group in year:
#     abc = group.groupby('Layout')
#     for name1, group1 in abc:
#         print(str(name),str(name1),len(group1))
#     print()

# 不同Year下的房屋总价的均值对比(可绘图，也可以用数字呈现)
# for name, group in year:
    # print(str(name)+ '年总价均值：',group['Price'].mean())

# 不同Year下的房屋总面积均值对比
# for name, group in year:
    # print(str(name)+ '年房屋总面积：', np.sum(group['Size']))

# 不同Year下的带和不带花园的房屋总数对比
# for name, group in year:
#     num = group.groupby('Renovation')
    # for name1, group1 in num:
        # print(str(name)+'年'+str(name1)+'总数：',len(group1))
    # print()


# 08 小区房屋特征分析
# 查询房屋平均单价最便宜和最贵的小区名
# df = df.loc[(df['PerPrice'] > 0) & (df['PerPrice'] < 20)]
# grouped = df.groupby('Garden')
# avg = {}
# for name,group in grouped:
#     avg[name] = group['PerPrice'].median()
# data = pd.Series(avg)
# data = data.sort_values()
# min = data[0]
# max = data[-1]
# print('房屋平均单价最便宜的小区名', min)
# print('房屋平均单价最贵的小区名', max)

# 查询房屋平均年份最老和最新的区县名
# grouped2 = df.groupby('Region')
# avg2 = {}
# for name, group in grouped2:
#     avg2[name] = group['Year'].median()
# 转为Series
# data2 = pd.Series(avg2)
# 通过value进行排序
# data2 = data2.sort_values()
# 最小值
# min_code = data2.keys()[0]
# print('房屋平均年份最老的区县名',min_code)
# 最大值
# max_code = data2.keys()[-1]
# print('房屋平均年份最新的区县名',max_code)

# 09 小区特征聚类分析
# 对提供数据中的所有小区进行聚类，聚类个数可按自已的想法自定义，聚类完成后打印出同一个类别中所有房屋的平均单价、平均面积、平均楼层信息。
# 提示：1、对房屋数据中非数字类别的特征可先转换为数字，再进行聚类；同一个特征转换的数字相同。比如 "有电梯"转换为1，"无电梯"转换为0
#       2、聚类的代码在之前提供的聚类demo代码中修改
a = []
b = []
hh = []
a.append(df['Direction'].unique())
for i in a:
    i
for j in i:
    if len(j) < 3:
        b.append(j)
for j in i:
    if len(j) > 2:
        hh.append(j)
for i in hh:
    df.loc[(df['Direction'] == i), 'Direction'] = random.randint(0, 10)
o = 1
for k in b:
    df.loc[df['Direction'] == k, 'Direction'] = o
    o = o + 1


c = []
c.append(df['District'].unique())
for i in c:
    i
o = 1
for k in i:
    df.loc[df['District'] == k, 'District'] = o
    o = o + 1


df.loc[df['Elevator'] == '有电梯', 'Elevator'] = 1
df.loc[df['Elevator'] == '无电梯', 'Elevator'] = 0


d = []
d.append(df['Garden'].unique())
for i in d:
    i
o = 1
for k in i:
    df.loc[df['Garden'] == k, 'Garden'] = o
    o = o + 1


e = []
e.append(df['Layout'].unique())
for i in e:
    i
o = 1
for k in i:
    df.loc[df['Layout'] == k, 'Layout'] = o
    o = o + 1


f = []
f.append(df['Region'].unique())
for i in f:
    i
o = 1
for k in i:
    df.loc[df['Region'] == k, 'Region'] = o
    o = o + 1


g = []
g.append(df['Renovation'].unique())
for i in g:
    i
o = 1
for k in i:
    df.loc[df['Renovation'] == k, 'Renovation'] = o
    o = o + 1


ff = [i for i in range(0,23656)]
gg = df['Garden']
plt.figure()
plt.scatter(gg,ff)

#聚类并画出分堆图
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 3) #分k堆
kmeans.fit(df)  #训练数据
cluster_label = kmeans.labels_
plt.figure()
plt.scatter(gg, ff, c=cluster_label)
plt.title('Kmeans K=' + str(3))
plt.show()

