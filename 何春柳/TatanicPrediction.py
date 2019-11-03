import pandas as pd
import numpy as np
#导入数据
Titanic = pd.read_csv('train.csv')
test1 = pd.read_csv('test.csv')
test = pd.read_csv('test.csv')

#删除无意义的变量，并检查剩余的字是否有缺失值
Titanic.drop(['PassengerId','Name','Ticket','Cabin'], axis = 1, inplace = True)
Titanic.isnull().sum(axis = 0)
test.drop(['PassengerId','Name','Ticket','Cabin'], axis = 1, inplace = True)
test.isnull().sum(axis = 0)

#对Sex分组，用各组乘客的平均年龄填充各组中的缺失年龄
fillna_Titanic = []
for i in Titanic.Sex.unique():
    update = Titanic.loc[Titanic.Sex ==i,].fillna(value={'Age':Titanic.Age[Titanic.Sex ==i].mean()},inplace=False)
    fillna_Titanic.append(update)
Titanic = pd.concat(fillna_Titanic)

fillna_Titanic1 = []
for i in test.Sex.unique():
    update =test.loc[test.Sex ==i,].fillna(value={'Age':test.Age[test.Sex ==i].mean()},inplace=False)
    fillna_Titanic1.append(update)
test = pd.concat(fillna_Titanic1)

#使用Embarked的众数填充缺失值
Titanic.fillna(value={'Embarked':Titanic.Embarked.mode()[0]},inplace=True)
Titanic.head()

test.fillna(value={'Embarked':test.Embarked.mode()[0]},inplace=True)
test.head()

## 将数值型的Pclass转换为类别型，否则无法对其哑变量处理
Titanic.Pclass = Titanic.Pclass.astype('category')
test.Pclass = test.Pclass.astype('category')
#哑变量处理
dummy = pd.get_dummies(Titanic[['Sex','Embarked','Pclass']])
dummy1 = pd.get_dummies(test[['Sex','Embarked','Pclass']])
#水平合并Titanic和哑变量的数据集
Titanic = pd.concat([Titanic,dummy],axis=1)
test = pd.concat([test,dummy1],axis=1)
#删除原始的Sex、Embarked、Pclass
Titanic.drop(['Sex','Embarked','Pclass'],inplace = True,axis = 1)
Titanic.head()
test.drop(['Sex','Embarked','Pclass'],inplace = True,axis = 1)


#收集数据
train_data = Titanic
test_data = test

#数据准备
train_y = train_data.Survived
train_X = train_data[['Age', 'SibSp', 'Parch', 'Fare', 'Sex_female', 'Sex_male',
                'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Pclass_1', 'Pclass_2',
                'Pclass_3']]

train_X.to_csv('Train_data.csv',index = False)
test_X = test_data[['Age', 'SibSp', 'Parch', 'Fare', 'Sex_female',
                    'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S',
                    'Pclass_1', 'Pclass_2', 'Pclass_3']]

print(test_X.isnull().sum(axis = 0)>0)
for column in list(test_X.columns[test_X.isnull().sum() > 0]):
    mean_val =test_X[column].mean()
    test_X[column].fillna(mean_val, inplace=True)
#选择一个模型
from sklearn import tree
clf = tree.DecisionTreeClassifier()

#训练
clf.fit(train_X,train_y)

#评估
acc = clf.score(train_X, train_y)
#print('训练集准确率：'+str(acc))


#预测
predictions = clf.predict(test_X)
result = pd.DataFrame({'PassengerId':test1['PassengerId'].as_matrix(),'Survived':predictions.astype(np.int32)})
result.to_csv('Decision_preditions.csv',index = False)
