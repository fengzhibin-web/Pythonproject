import pandas as pd
# 使用 pd.read_csv 方法读取 CSV 文件
iris = pd.read_csv(r'D:\Pythonproject\dataanalysis_iris\iris.csv')

# 数据浏览：
print('空值数量统计：')
print(iris.isnull().sum())

print("\n浏览数据前 5 行：")
print(iris.head())  # 使用 head() 默认值查看前五行

print("\n浏览数据前 10 行：")
print(iris.head(10))    # 使用 head(n) 查看前 n 行

print("\n浏览数据后 5 行：")
print(iris.tail())  # 使用 tail() 默认值查看后五行

# 获取数据的基本信息，包括每列的非空值数目、数据类型等
print("\n数据的基本信息：")
iris.info()  # info() 方法显示数据框的结构

# 数据描述性统计，包括均值、标准差、最小值、四分位数等
print("\n数据集描述性统计：")
print(iris.describe())  # describe() 方法查看描述性统计

# 数据选择：
# 选择特定的列
print("\n选择 'sepal_length' 和 'sepal_width' 两列数据前五行：")
print(iris[['sepal_length', 'sepal_width']].head())  # 选择并查看特定的两列数据前五行

# 条件选择：基于特定条件过滤数据
print("\n选择 sepal_length 大于 5 的数据：")
print(iris[iris['sepal_length'] > 5].head())  # 选择 sepal_length 大于 5 的行

# 简单的数据统计操作：
# 计算 sepal_length 列的均值
print("\nsepal_length 的均值：")
print(iris['sepal_length'].mean())  # 计算 sepal_length 列的均值

# 计算 sepal_width 列的总和
print("\nsepal_width 的总和：")
print(iris['sepal_width'].sum())  # 计算 sepal_width 列的总和

# 计算数据集中某列的非空值数量
print("\nsepal_length 列的非空值数量：")
print(iris['sepal_length'].count())  # 计算 sepal_length 列的非空值数量

# 其他常见的数据操作：
# 查看唯一值
print("\n查看 'species' 列中的唯一值：")
print(iris['species'].unique())  # 查看 species 列中的唯一值

# 查看种类数量
print("\n鸢尾花种类的个数：")
print(iris['species'].nunique())

# 查看每个类别的数量
print("\n每个类别鸢尾花的数量：")
print(iris['species'].value_counts())  # 查看 species 列中每个类别的数量