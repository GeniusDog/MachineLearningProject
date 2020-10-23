'''
    描述：足球运动员的特征分析小项目代码部分
'''

# 导入分析需要用到的库
import numpy as np
import pandas as pd
import matplotlib as mpl
import mptplotlib.pyplot as plt

# 设置 matplotlib 支持中文的显示
mpl.rcParams["font.family"]="SimHei"

# 设置 matplotlib 支持负号的显示
mpl.rcParams["axes.unicode_minus"]=False

# pandas 加载数据集FullData.csv,数据对象为player
player=pd.read_csv(r"FullData.csv")

# 显示当前所有列的数据集
pd.set_option("max_columns",100)
player.head()

# 对数据集进行简单的数据查看：如缺失信息等
# 通过info方法查看缺失信息（以及每列的类型信息）
player.info()

# 使用player对 Club_Position 属性这行非空的数据进行过滤，过滤完再传回给player
player=player[player["Club_Position"].notnull()]
player.info()

# 查看异常值
player.describe()

# 查看是否包含重复值
player.duplicated().any()

# 如果包含重复值，调用下面这条语句,inplace=True 返回的结果在player上执行，不会生成新的对象
# player.drop_duplicates(inplace=True)

# 再次查看player对象，开始观察数据
player.head()

# 将 Height 中的cm替换成空格
player["Height"]=player["Height"].map(lambda x:int(x.replace("cm","")))
# player["Height"]=player["Height"].str.replace("cm","").astype(np.int)

# 将 Weight 中的kg替换成空格
player["Weight"]=player["Weight"].map(lambda x:int(x.replace("kg","")))
# player["Weight"]=player["Weight"].str.replace("kg","").astype(np.int)


# 再次查看player对象
player.head()

# 查看运动员身高，体重，评分信息的分布图：核密度估计图kde
player[["Height","Weight","Rating"]].plot(kind="kde")

# 足球运动员左脚和右脚上的偏差：统计情况
player["Preffered_Foot"].value_counts()

# 将上面左右脚的统计结果进行画图:条形图bar
player["Preffered_Foot"].value_counts().plot(kind="bar")

# 从球员的平均评分考虑:先根据Club分组，再获取Rating，然后对分组的内容进行计算
s=player.groupby("Club")["Rating"].agg(["count","sum","mean"])

# 对结果s进行过滤，只需要前10的内容
s=s[s["count"]>10]

# 根据count将s进行降序排序,显示前十位的结果
s.sort_values("count",ascending=False).head(10)

# 根据mean将s进行降序排序,显示前十位的结果
s.sort_values("mean",ascending=False).head(10)

# 从球员的平均评分考虑:先根据Nationality分组，再获取Rating，然后对分组的内容进行计算
s_Nationality=player.groupby("Nationality")["Rating"].agg(["count","sum","mean"])

# 对结果s进行过滤，只需要前10的内容
s_Nationality=s_Nationality[s_Nationality["count"]>10]

# 根据mean将s进行降序排序,显示前十位的结果
s_Nationality.sort_values("mean",ascending=False).head(10)

# 哪个俱乐部拥有5年及其以上的球员
# 获取年份,并转化成数值类型
year=player["Club_Joining"].map(lambda x:int(x.split("/")[-1]))
# 除了用int()转化，也可以使用np.int转化
# year=year.astype(np.int)

# 获取5年以上球员的信息
t=player[(2020-year) & (player["Club"] !="Free Agents")]

# 将结果显示出来
t["Club"].value_counts().head(10).plot(kind="bar")

# 全体运动员的出生月份与评分是否有关系
y=player["Birth_Date"].str.split("/",expand=True)
y[0].value_counts().plot(kind="bar")

# 知名运动员（评分80分以上为知名）的出生月份与评分是否有关系
y_perfect=player[player["Rating"]>=80]
y_perfect=player["Birth_Date"].str.split("/",expand=True)
y_perfect[0].value_counts().plot(kind="bar")

# 足球运动员的号码是否与位置有关
# 去除替补球员和后备队球员
t=player[(player["Club_Position"] != "Sub" ) & (player["Club_Position"] !="Res")]

# 为了便于观察，将对应球员俱乐部的号码和位置进行分组
x=t.groupby(["Club_Kit","Club_Position"]).size()
x[x>50].plot(kind="bar")

# 足球运动员的身高和体重是否有关联，散点图查看
player.plot.scatter(x="Height",y="Weight")

# 采用相关系数的方法，来观察哪些指标对数据集的影响大
player.corr()

# 假设不知到数据集中后两行的含义，由数据集的特征来推断GK_Handling
#player.info()

# 现根据 Club_Position 对其进行分组
t=player.groupby("Club_Position")

# 获取分组后GK_Handling的均值情况，条形图展示
t["GK_Handling"].agg("mean").plot(kind="bar")

# 假设不知到数据集中后两行的含义，由数据集的特征来推断GK_Reflexes
#player.info()

# 现根据 Club_Position 对其进行分组
t=player.groupby("Club_Position")

# 获取分组后GK_Handling的均值情况，条形图展示
t["GK_Reflexes"].agg("mean").plot(kind="bar")

# 观察年龄与评分之间的关系
t=player[["Age","Rating"]]

# 使用pandas将年龄Age进行离散化
t["Age"]=pd.cut(player["Age"],bins=[0,20,30,40,60],labels=["小","中","大","很大"])

# 根据t的年龄进行分组，得到评分的均值
t.groupby("Age")["Rating"].mean().plot(kind="line",xticks=[0,1,2,3,4],marker="o")
