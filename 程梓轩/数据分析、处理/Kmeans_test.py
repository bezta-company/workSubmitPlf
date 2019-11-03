#########################################
# CODE:   Kmeans 聚类 Pandas实现示例代码 #
# AUTHOR: Tiger                         #
# TIME:   2019-09-25                    #
#########################################
import numpy as np
import pandas as pd
import sys
import sklearn.datasets as dt

# headers = {"IRIS": list(range(0, 4)), "BOSTON": list(range(0, 13)), "WINE": list(range(0, 13))}

class Kmeans:
    def __init__(self, clusterNum, center, isCenterDefined, data,headers):#传参的
        self.clusterNum = clusterNum
        self.centerSample = center
        self.isCenterDefined = isCenterDefined
        self.data = data
        self.headers = headers
    def getClusterData(self):
        data = self.data
        headers = self.headers
        #读取数据后，需要再作一次归一化(聚类为何要归一化，自已百度)
        clusterData = pd.DataFrame(data, columns = headers).apply(lambda line: (line - min(line))/(max(line) - min(line)), axis = 0)
        clusterData['CLUSTER'] = np.nan
        return clusterData

    def getCurCenter(self, loop, data):
        clusterNum = self.clusterNum
        headers = self.headers
        if loop < 0:
            raise Exception("Loop parameter can not less than 0")
        if len(data) == 0:
            raise Exception("Data to be clustered are empty")
        # 若是首轮循环，判断初始聚类点生成方式
        if loop == 0:
            # 自定义
            if self.isCenterDefined == 1:
                if len(self.centerSample) != clusterNum:
                    raise Exception("Center points num are not equal to clusterNum")
                new_center_point = data.iloc[self.centerSample, headers] #根据手动指定的行索引来选择聚类中心数据
            # 不定义，采用随机方式生成
            else:
                np.random.seed()
                new_center_point = data.iloc[np.random.randint(0, len(data), clusterNum), headers]#根据指定的聚类数来随机生成聚类中心数据
            return new_center_point
        # 非首轮循环，计算中心点生成方式
        else:
            if data['CLUSTER'].value_counts().count() != self.clusterNum:
                raise Exception('Cluster num is not equal to %d' % self.clusterNum)
            index = 0
            for center_label in data['CLUSTER'].unique():
                new_center_point = pd.DataFrame(data.loc[data.CLUSTER == center_label, headers].mean())
                if index == 0:
                    center_points = new_center_point
                    index = index + 1
                else:
                    center_points = pd.concat([center_points,new_center_point],axis = 1)
            return center_points.T

    def gernerateNewCluster(self, sample, center_points):
        headers = self.headers
        if len(center_points) != self.clusterNum:
            raise Exception('Center points num is not equal to cluster num')
        point_dist = []
        for line_num in range(0, len(center_points)):
            point_dist.append(np.sum(np.square(np.array(sample[headers[self.fileName]]) - np.array(center_points.iloc[line_num,:]))))
        min_Dist = min(point_dist)
        # 求最短距离处对应的中心点索引，索引值默认由中心点的位置顺序值决定
        return pd.Series([point_dist.index(min_Dist), min_Dist], index = ['CLUSTER','DISTANCE'])
    def demand(self,maxLoop,threshold):
        self.maxLoop = maxLoop
        self.threshold = threshold
        self.process()

    def process(self):
        maxLoop = self.maxLoop
        threshold = self.threshold
        data = self.data
        last_dist = 0
        try:
            #读取初始待聚类的数据
            clusterDt = self.getClusterData()
            for loop in range(0, maxLoop):
                if loop % 5 == 0:
                    print('Loop times:%d' % loop)
                #计算当前迭代中的簇中心点
                curCenter = self.getCurCenter(loop, clusterDt)
                #计算当前迭代中每条样本所归属的簇标签
                result = clusterDt.apply(lambda line: Kmeans.gernerateNewCluster(line, curCenter), axis = 1, result_type = 'expand')
                clusterDt['CLUSTER'] = result['CLUSTER']
                #计算当前轮迭代完成后所有簇中各样本与中心点间的距离之和
                cur_dist = result['DISTANCE'].sum()
                #计算前后两次迭代中，样本距中心点距离之和的变化量，若变化量趋近于0，则聚类结果趋于稳定
                cur_dist_threshold = abs(cur_dist - last_dist)
                #若变化量小于自定义的门限，则聚类迭代停止
                if cur_dist_threshold <= threshold:
                    print('Current cluster centers are:\n', curCenter)
                    print("Cluster process end with the satisfication of threshold %f" % cur_dist_threshold)
                    exit(0)
                else:
                    print('Current cluster average distance:%f' % cur_dist_threshold)
                    last_dist = cur_dist
                #若迭代达到最大自定义轮数，则迭代停止
                if loop == (maxLoop - 1):
                    print('Achieved Max Loop Times:%d' % loop)
                    print('Current cluster centers are:\n', curCenter)
        except Exception as e:
            print('Error occur: %s' % e.args)


if __name__ == "__main__":
    A = Kmeans
# if __name__ == '__main__':
#     demo = Demo()
#     demo.test()



