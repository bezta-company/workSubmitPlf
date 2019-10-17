#########################################
# CODE:   Python数据分析示例代码         #
# AUTHOR: Tiger                         #
# TIME:   2019-09-28                    #
#########################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
lianjia_df = pd.read_csv("lianjia.csv")
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
print(lianjia_df.describe())
#    列值类型总览
print(lianjia_df.info())

# 01 新增列特征、重排数据特征的列顺序
df = lianjia_df.copy()
df['PerPrice'] = lianjia_df['Price'] / lianjia_df['Size']
columns = ['Region', 'District', 'Garden', 'Layout', 'Floor', 'Year', 'Size', 'Elevator', 'Direction', 'Renovation', 'PerPrice', 'Price']
df = pd.DataFrame(df, columns = columns)

#02不同建房年份的特征
#不同年份的房屋户型的数量对比

house_count1= df.groupby(by = 'Year')['Layout'].value_counts()

# 不同Year下的房屋总价的均值对比
Price_mean=df.groupby(by='Year')['Price'].mean().sort_values(ascending=False).to_frame().reset_index()
# 不同Year下的房屋总面积均值对比
Size_mean=df.groupby(by='Year')['Size'].mean().sort_values(ascending=False).to_frame().reset_index()

f,[plt1,plt2]=plt.subplots(1,2,figsize=(10,20))
sns.barplot(x = 'Year', y = 'Price', palette = "Blues_d", data = Price_mean, ax = plt1)
plt1.set_title('北京各年份总价均值对比', fontsize = 12)
plt1.set_xlabel(' ')
sns.barplot(x='Year',y='Size',palette="Greens_d",data=Size_mean,ax=plt2)
plt2.set_title('北京各年份房屋总面积均值对比',fontsize=12)
plt2.set_xlabel(' ')
plt.show()
# 不同Year下的带和不带花园的房屋总数对比


# 03小区房屋特征分析
# 查询房屋平均单价最便宜和最贵的小区名
PerPrice_mean=df.groupby(by='Garden')['PerPrice'].mean().sort_values(ascending=False).to_frame()

max_mean=PerPrice_mean.max()
min_mean=PerPrice_mean.min()
print(max_mean,min_mean)


# 查询房屋平均年份最老和最新的区县名

Year_mean=df.groupby(by='District')['Year'].mean().sort_values(ascending=False)

# 09 小区特征聚类分析
# 对提供数据中的所有小区进行聚类，聚类个数可按自已的想法自定义，聚类完成后打印出同一个类别中所有房屋的平均单价、平均面积、平均楼层信息.
df.loc[(df['Floor'] > 6) & (df['Elevator'].isnull()), 'Elevator'] = '有电梯'
df.loc[(df['Floor'] <= 6) & (df['Elevator'].isnull()), 'Elevator'] = '无电梯'

df.loc[df['Elevator']=='有电梯','Elevator']=1
df.loc[df['Elevator']=='无电梯','Elevator']=0

df.loc[df['Floor'].isnull(), 'Floor'] = 10

df.loc[df['Renovation']=='简装', 'Renovation'] = 2
df.loc[df['Renovation']=='精装', 'Renovation'] = 3

df.loc[(df['Renovation'] =='其他') | (df['Renovation'].isnull())  | (df['Renovation'] =='毛坯'), 'Renovation'] = 1
df1= df.drop(['Direction','District','Garden','Layout','Region'],axis=1).head(20)
df2 = df1.astype(int)

class Kmeans:
    def __init__(self, fileName, clusterNum, center, isCenterDefined):
        self.fileName = fileName
        self.clusterNum = clusterNum
        self.centerSample = center
        self.isCenterDefined = isCenterDefined

    def getClusterData(self):

            data = self.fileName

            # 读取数据后，需要再作一次归一化(聚类为何要归一化，自已百度)
            clusterData = data.apply(lambda line: (line - min(line)) / (max(line) - min(line)), axis=0)
            clusterData['CLUSTER'] = np.nan
            return clusterData

    def getCurCenter(self, loop, data):
        if loop < 0:
            raise Exception("loop parameter can not less than 0")
        if len(data) == 0:
            raise Exception("data to be clustered are empty")
        # 若是首轮循环，判断初始聚类点生成方式
        if loop == 0:
            # 自定义
            if self.isCenterDefined == 1:
                if len(self.centerSample) != self.clusterNum:
                    raise Exception("center points num are not equal to clusterNum")
                new_center_point = data.iloc[self.centerSample,self.fileName]  # 根据手动指定的行索引来选择聚类中心数据
            # 不定义，采用随机方式生成
            else:
                np.random.seed()
                new_center_point = data.iloc[
                    np.random.randint(0, len(data), self.clusterNum), self.fileName]  # 根据指定的聚类数来随机生成聚类中心数据
            return new_center_point
        # 非首轮循环，计算中心点生成方式
        else:
            if data['CLUSTER'].value_counts().count() != self.clusterNum:  # .value_counts()返回series

                raise Exception('cluster num is not equal to %d' % self.clusterNum)
            index = 0
            for center_label in data['CLUSTER'].unique():
                new_center_point = pd.DataFrame(data.loc[data.CLUSTER == center_label, self.fileName].mean())
                if index == 0:  # 也可以用df.["列名"]
                    center_points = new_center_point
                    index = index + 1
                else:
                    center_points = pd.concat([center_points, new_center_point], axis=1)
            return center_points.T

    def gernerateNewCluster(self, sample, center_points):
        if len(center_points) != self.clusterNum:
            raise Exception('center points num is not equal to cluster num')
        point_dist = []
        for line_num in range(0, len(center_points)):
            point_dist.append(
                np.sum(np.square(np.array(sample[self.fileName]) - np.array(center_points.iloc[line_num, :]))))
        min_Dist = min(point_dist)
        # 求最短距离处对应的中心点索引，索引值默认由中心点的位置顺序值决定
        return pd.Series([point_dist.index(min_Dist), min_Dist], index=['CLUSTER', 'DISTANCE'])


if __name__ == "__main__":

    func = Kmeans

    last_dist = 0
    try:
        # 读取初始待聚类的数据
        clusterDt = func.getClusterData()
        for loop in range(0, 400):
            if loop % 5 == 0:
                print('Loop times:%d' % loop)
            # 计算当前迭代中的簇中心点
            curCenter = func.getCurCenter(loop, clusterDt)
            # 计算当前迭代中每条样本所归属的簇标签
            result = clusterDt.apply(lambda line: func.gernerateNewCluster(line, curCenter), axis=1,result_type='expand')
            clusterDt['CLUSTER'] = result['CLUSTER']
            # 计算当前轮迭代完成后所有簇中各样本与中心点间的距离之和
            cur_dist = result['DISTANCE'].sum()
            # 计算前后两次迭代中，样本距中心点距离之和的变化量，若变化量趋近于0，则聚类结果趋于稳定
            cur_dist_threshold = abs(cur_dist - last_dist)  # abs绝对值
            # 若变化量小于自定义的门限，则聚类迭代停止
            if cur_dist_threshold <= 0.0004:
                print('current cluster centers are:\n', curCenter)
                print("cluster process end with the satisfication of threshold %f" % cur_dist_threshold)
                exit(0)
            else:
                print('current cluster average distance:%f' % cur_dist_threshold)
                last_dist = cur_dist
            # 若迭代达到最大自定义轮数，则迭代停止
            if loop == (400 - 1):
                print('Achieved Max Loop Times:%d' % loop)
                print('current cluster centers are:\n', curCenter)
    except Exception as e:
        print('error occur')

############   以下为本次数据分析的作业   ################

# 07 Year 分析不同建房年份下的房屋特征
# 不同Year下的各房屋户型的数量对比(可以绘图，也可以用数字呈现；比如：2000年，一室一厅，20；2000年，两室一厅，30）
# 不同Year下的房屋总价的均值对比(可绘图，也可以用数字呈现)
# 不同Year下的房屋总面积均值对比
# 不同Year下的带和不带花园的房屋总数对比


# 08 小区房屋特征分析
# 查询房屋平均单价最便宜和最贵的小区名
# 查询房屋平均年份最老和最新的区县名

# 09 小区特征聚类分析
# 对提供数据中的所有小区进行聚类，聚类个数可按自已的想法自定义，聚类完成后打印出同一个类别中所有房屋的平均单价、平均面积、平均楼层信息。
# 提示：1、对房屋数据中非数字类别的特征可先转换为数字，再进行聚类；同一个特征转换的数字相同。比如 "有电梯"转换为1，"无电梯"转换为0
#       2、聚类的代码在之前提供的聚类demo代码中修改