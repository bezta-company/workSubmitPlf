#########################################
# CODE:   Kmeans 聚类 Pandas实现示例代码 #
# AUTHOR: Tiger                         #
# TIME:   2019-09-25                    #
#########################################
import numpy as np
import pandas as pd
import sys
# import sklearn.datasets as dt

# headers = {"IRIS": list(range(0, 4)), "BOSTON": list(range(0, 13)), "WINE": list(range(0, 13))}



class Kmeans:
    def __init__(self, fileName, clusterNum, center, isCenterDefined, data):#传参的
        self.fileName = fileName
        self.clusterNum = clusterNum
        self.centerSample = center
        self.isCenterDefined = isCenterDefined
        self.data = data
        headers = data.columns.values.tolist()
    def getClusterData(self):
        # try:
        #     if self.fileName == 'IRIS':
        #         data, label = dt.load_iris(True)
        #     elif self.fileName == 'BOSTON':
        #         data, label = dt.load_boston(True)
        #     else:
        #         data, label = dt.load_wine(True)
        # except Exception as e:
        #     print(e.strerror)
        data,label = self.data(True)
        headers = data.columns.values.tolist()
        #读取数据后，需要再作一次归一化(聚类为何要归一化，自已百度)
        clusterData = pd.DataFrame(data, columns = headers[self.fileName]).apply(lambda line: (line - min(line))/(max(line) - min(line)), axis = 0)
        clusterData['CLUSTER'] = np.nan
        return clusterData

    def getCurCenter(self, loop, data, clusterNum):
        headers = data.columns.values.tolist()
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
                new_center_point = data.iloc[self.centerSample, headers[self.fileName]] #根据手动指定的行索引来选择聚类中心数据
            # 不定义，采用随机方式生成
            else:
                np.random.seed()
                new_center_point = data.iloc[np.random.randint(0, len(data), clusterNum), headers[self.fileName]]#根据指定的聚类数来随机生成聚类中心数据
            return new_center_point
        # 非首轮循环，计算中心点生成方式
        else:
            if data['CLUSTER'].value_counts().count() != self.clusterNum:
                raise Exception('Cluster num is not equal to %d' % self.clusterNum)
            index = 0
            for center_label in data['CLUSTER'].unique():
                new_center_point = pd.DataFrame(data.loc[data.CLUSTER == center_label, headers[self.fileName]].mean())
                if index == 0:
                    center_points = new_center_point
                    index = index + 1
                else:
                    center_points = pd.concat([center_points,new_center_point],axis = 1)
            return center_points.T

    def gernerateNewCluster(self, sample, center_points, data):
        headers = data.columns.values.tolist()
        if len(center_points) != self.clusterNum:
            raise Exception('Center points num is not equal to cluster num')
        point_dist = []
        for line_num in range(0, len(center_points)):
            point_dist.append(np.sum(np.square(np.array(sample[headers[self.fileName]]) - np.array(center_points.iloc[line_num,:]))))
        min_Dist = min(point_dist)
        # 求最短距离处对应的中心点索引，索引值默认由中心点的位置顺序值决定
        return pd.Series([point_dist.index(min_Dist), min_Dist], index = ['CLUSTER','DISTANCE'])

def rukou(dataName, maxLoop, clusterNum, center, isCenterDefined,threshold, data):
    if __name__ == "__main__":
        # if len(sys.argv) != 7:
        #     raise Exception('Argument num is %n , but requred num is 6' % len(sys.argv))
        # def _new_(dataName, maxLoop, clusterNum, center, isCenterDefined, threshold):

        # try:
        #     #通过命令行传入所需要的参数
        #     #argv[1] -- 聚类数据名,可选
        #     #argv[2] -- 最大迭代次数
        #     #argv[3] -- 聚类个数
        #     #argv[4] -- 自定义的初始中心点对应的样本行号
        #     #argv[5] -- 是否使用自定义中心点 1 -- 使用 0 -- 不使用
        #     #argv[6] -- 相邻两次聚类前后簇内平均样本间距的变化量
        #     dataName = sys.argv[1]
        #     maxLoop = int(sys.argv[2])
        #     clusterNum = int(sys.argv[3])
        #     center = [int(re) for re in sys.argv[4].split(sep = ',')]
        #     isCenterDefined = int(sys.argv[5])
        #     threshold = float(sys.argv[6])
        #     data = pd.DataFrame(sys.argv[7])
        # except Exception as e:
        #     raise(e.args)
        #     dataName = dataName
        #     maxLoop = maxLoop
        # clusterNum = clusterNum
        # center = center
        # isCenterDefined = isCenterDefined
        # threshold = threshold
        # data = data









        #初始化聚类class对象
        func = Kmeans(dataName, clusterNum, center, isCenterDefined, data)
        last_dist = 0
        try:
            #读取初始待聚类的数据
            clusterDt = func.getClusterData()
            for loop in range(0, maxLoop):
                if loop % 5 == 0:
                    print('Loop times:%d' % loop)
                #计算当前迭代中的簇中心点
                curCenter = func.getCurCenter(loop, clusterDt)
                #计算当前迭代中每条样本所归属的簇标签
                result = clusterDt.apply(lambda line: func.gernerateNewCluster(line, curCenter), axis = 1, result_type = 'expand')
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




