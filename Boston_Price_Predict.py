# 时间：2020.10.27 13点31分
# 任务：波士顿房价预测
# 目标：给定某地区的特征

# 导入必要的库
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import sklearn.datasets

# 设置 matplotlib 支持中文的显示
plt.rcParams["font.family"]="SimHei"

# 设置 matplotlib 支持负号的显示
plt.rcParams["axes.unicode_minus"]=False

#显示所有的列
pd.set_option('display.max_columns', None)

#显示所有的行
pd.set_option('display.max_rows', None)

#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)
# 获取需要分析的数据集,注意这里的load_boston()有括号，不然会报错
boston_hoursing=sklearn.datasets.load_boston()

# 导入数据集中的所有特征变量
boston_feature=boston_hoursing.data
#导入特征名
boston_feature_name = boston_hoursing.feature_names

# 导入数据集中的标签，也就是目标值：房价
boston_target=boston_hoursing.target

# 使用pandas设置要画出的内容框架:数据，列显示，标签值
boston_hoursing_show=pd.DataFrame(boston_feature)
boston_hoursing_show.columns=boston_feature_name
boston_hoursing_show["Price"]=boston_target
boston_hoursing_show.head()

# 绘制价格与住宅面积的散点图
plt.scatter(boston_hoursing_show.ZN,boston_hoursing_show.Price)
plt.xlabel("住宅面积")
plt.ylabel("住房价格")

# 绘制价格与平均房间数的散点图
plt.scatter(boston_hoursing_show.RM,boston_hoursing_show.Price)
plt.xlabel("平均房间数")
plt.ylabel("住房价格")

# 绘制价格与到就业中心的加权平均距离的散点图
plt.scatter(boston_hoursing_show.DIS,boston_hoursing_show.Price)
plt.xlabel("到就业中心的加权平均距离")
plt.ylabel("住房价格")

# 绘制价格与高速公路的便利指数的散点图
plt.scatter(boston_hoursing_show.RAD,boston_hoursing_show.Price)
plt.xlabel("高速公路的便利指数")
plt.ylabel("住房价格")

# 绘制价格与犯罪率的散点图
plt.scatter(boston_hoursing_show.CRIM,boston_hoursing_show.Price)
plt.xlabel("犯罪率")
plt.ylabel("住房价格")

# 异常值处理：通过观测有16个目标值值为50.0的数据点需要被移除
i_=[]
for i in range(len(boston_target)):
    if boston_target[i] == 50:
        i_.append(i)#存储房价等于50 的异常值下标
        
boston_feature = np.delete(boston_feature,i_,axis=0)#删除房价异常值数据
boston_target = np.delete(boston_target,i_,axis=0)#删除异常值
print("当前数据集中的特征数据的行数以及列数为：",np.shape(boston_feature))
print("当前数据集中的标签数据的行数为：",np.shape(boston_target))

# 将数据划分为测试机和训练集,测试样本30%
from sklearn.model_selection import train_test_split
boston_feature_train,boston_feature_test,boston_target_train,boston_target_test=train_test_split(boston_feature,
                                                                                                 boston_target,
                                                                                                 random_state=0,
                                                                                                 test_size=0.30)

# 1.导入线性回归
from sklearn.linear_model import LinearRegression

# 2.创建模型：线性回归
boston_model1 = LinearRegression()

# 3.训练模型
boston_model1.fit(boston_feature_train,boston_target_train)

# 训练后的截距
a1 = boston_model1.intercept_
# 训练后的回归系数
b1 = boston_model1.coef_
print('最佳拟合线：截距a = ',a1,'\n回归系数b = ',b1)

# 训练数据的预测值
boston_feature_predict1=boston_model1.predict(boston_feature_train)
for i, prediction in enumerate(boston_feature_predict1):
    print('Predicted: %s, Target: %s' % (prediction, boston_target_test[i]))

# 模型评估
accurucy_1=boston_model1.score(boston_feature_test , boston_target_test)
print("模型的准确性为：",accurucy_1)

# 逻辑回归
# 1.导入逻辑回归
from sklearn.linear_model import LogisticRegression

# 2.创建模型：逻辑回归
boston_model2 = LogisticRegression()

# 3.数据处理
for i in range(len(boston_feature_train)):
    print("boston_feature_train:",i,"boston_feature_train[i]:",boston_target_train[i])
    for j in range(len(boston_feature_train[i])):
        boston_feature_train[i][j]=int(boston_feature_train[i][j]*1000)
for i in range(len(boston_target_train)):
    print("boston_target_train:",i,"boston_target_train[i]:",boston_target_train[i])
    boston_target_train[i]=int(boston_target_train[i]*1000)
    
# 4.模型训练
boston_model2.fit(boston_feature_train,boston_target_train)

# 评估模型的准确率
accurucy_2=boston_model2.score(boston_feature_test,boston_target_test)
print("模型的准确性为：",accurucy_2)