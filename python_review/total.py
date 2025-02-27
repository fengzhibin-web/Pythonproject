print("1.基本变量和数据类型演示:")
# python基本变量和数据类型演示
a = 10  # 整数类型
b = 3.14  # 浮点数类型
name = "Alen"  # 字符串类型
is_active = True  # 布尔类型
fruits = ["apple", "banana", "cherry"]  # 列表（List）
person = {"name": "John", "age": 25}    # 字典（Dictionary）
colors = {"red", "green", "blue"}   # 集合（Set）

# 打印不同类型的变量并输出它们的数据类型
print(a, type(a))  # 输出整型
print(b, type(b))  # 输出浮点型
print(name, type(name))  # 输出字符串
print(is_active, type(is_active))  # 输出布尔值
print(fruits, type(fruits))  # 输出列表
print(person, type(person))  # 输出字典
print(colors, type(colors))  # 输出集合
print("_______________________________")
print("2.控制流演示:")
# if-else示例
x = 5
if x > 0:
    print("x is positive")
else:
    print("x is non-positive")

# for 循环示例
fruits = ["apple", "banana", "cherry"]
for fruit in fruits:
    print(fruit)#遍历输出列表元素

# while 循环示例
count = 0
while count < 5:
    print("Count:", count)
    count += 1
print("_______________________________")
print("3.函数定义和调用演示:")
# 定义函数
def greet(name):
    print(f"Hello, {name}!")

# 调用函数
greet("Alen")

# 定义带返回值的函数
def add(a, b):
    return a + b

# 调用带返回值的函数
result = add(10, 20)
print("Sum:", result)
print("_______________________________")
print("4.模块导入和调用演示:")
# 导入math模块
import math

# 使用 math 模块中的 sqrt 函数
result = math.sqrt(16)
print("Square root of 16:", result)

# 调用模块功能
from math import pi

# 使用导入的 pi
print("Value of pi:", pi)
print("_______________________________")
