import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
import numpy as np
# 初始化 rich 控制台
console = Console()

# 创建 data_pic 文件夹
os.makedirs('data_pic', exist_ok=True)

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取CSV文件
df = pd.read_csv('cifar10_recognition_data_batch_1.csv')

# 1. True_Label 分布
label_counts = df['True_Label'].value_counts().sort_index()
label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# 打印数据到控制台
console.print("[bold magenta]CIFAR-10 样本数量分布[/bold magenta]")
table = Table(show_header=True, header_style="bold blue")
table.add_column("标签", style="dim")
table.add_column("数量", justify="right")
for label, count in zip(label_names, label_counts):
    table.add_row(str(label), str(count))  # 将label和count都转换为字符串
console.print(table)

# 绘制图表
plt.figure(figsize=(10, 6))
bars = plt.bar(label_names, label_counts, color='skyblue')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, int(height), ha='center', va='bottom', fontsize=12)
plt.title('CIFAR-10 样本数量分布', fontsize=16)
plt.xlabel('标签', fontsize=14)
plt.ylabel('数量', fontsize=14)
plt.savefig('data_pic/true_label_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. 识别正确率分布
correct_counts = df['IsCorrect'].value_counts().sort_index(ascending=False)
correct_labels = [f'识别正确 ({correct_counts.iloc[0]})', f'识别错误 ({correct_counts.iloc[1]})']

# 打印数据到控制台
console.print("[bold magenta]识别正确率分布[/bold magenta]")
table = Table(show_header=True, header_style="bold blue")
table.add_column("状态", style="dim")
table.add_column("数量", justify="right")
for label, count in zip(correct_labels, correct_counts):
    table.add_row(label, str(count))
console.print(table)

# 绘制图表
plt.figure(figsize=(8, 5))
plt.pie(correct_counts, labels=correct_labels, autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'], startangle=90)
plt.title('识别正确率分布', fontsize=16)
plt.savefig('data_pic/recognition_accuracy.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. API 处理时间分布
processing_times = df['API Processing Time'].str.extract('(\d+\.\d+)')[0].astype(float)

# 打印数据到控制台
console.print("[bold magenta]API 处理时间分布[/bold magenta]")
table = Table(show_header=True, header_style="bold blue")
table.add_column("时间范围（秒）", style="dim")
table.add_column("数量", justify="right")
bins = pd.cut(processing_times, bins=10)
for bin_range, count in bins.value_counts().sort_index().items():
    table.add_row(str(bin_range), str(count))
console.print(table)

# 绘制图表
plt.figure(figsize=(10, 6))
sns.histplot(processing_times, bins=20, kde=True, color='orange')
plt.title('API 处理时间分布', fontsize=16)
plt.xlabel('处理时间（秒）', fontsize=14)
plt.ylabel('数量', fontsize=14)
plt.savefig('data_pic/api_processing_time.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. 识别效率评估：计算不同类型True_Label的平均识别时间并可视化
df['ProcessingTime'] = processing_times
avg_processing_time = df.groupby('True_Label')['ProcessingTime'].mean().reset_index()
avg_processing_time['LabelName'] = label_names

# 打印数据到控制台
console.print("[bold magenta]不同类型True_Label的平均识别时间[/bold magenta]")
table = Table(show_header=True, header_style="bold blue")
table.add_column("标签", style="dim")
table.add_column("平均处理时间（秒）", justify="right")
for _, row in avg_processing_time.iterrows():
    table.add_row(row['LabelName'], f"{row['ProcessingTime']:.2f}")
console.print(table)

# 绘制图表
plt.figure(figsize=(12, 6))
bars = sns.barplot(x='LabelName', y='ProcessingTime', hue='LabelName', data=avg_processing_time, palette='viridis', legend=False)
plt.title('不同类型True_Label的平均识别时间', fontsize=16)
plt.xlabel('标签', fontsize=14)
plt.ylabel('平均处理时间（秒）', fontsize=14)
plt.ylim(0, max(avg_processing_time['ProcessingTime']) * 1.1)
plt.xticks(rotation=45)
for bar in bars.patches:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}s', ha='center', va='bottom', fontsize=10)
plt.savefig('data_pic/avg_processing_time_by_label.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. 识别正确率与标签的关系
correct_rate_by_label = df.groupby('True_Label')['IsCorrect'].mean().reset_index()
correct_rate_by_label['LabelName'] = label_names

# 打印数据到控制台
console.print("[bold magenta]不同标签的识别正确率[/bold magenta]")
table = Table(show_header=True, header_style="bold blue")
table.add_column("标签", style="dim")
table.add_column("识别正确率", justify="right")
for _, row in correct_rate_by_label.iterrows():
    table.add_row(row['LabelName'], f"{row['IsCorrect']:.2%}")
console.print(table)

# 绘制图表
plt.figure(figsize=(12, 6))
bars = sns.barplot(x='LabelName', y='IsCorrect', hue='LabelName', data=correct_rate_by_label, palette='coolwarm', legend=False)
plt.title('不同标签的识别正确率', fontsize=16)
plt.xlabel('标签', fontsize=14)
plt.ylabel('识别正确率', fontsize=14)
plt.xticks(rotation=45)
for bar in bars.patches:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2%}', ha='center', va='bottom', fontsize=10)
plt.savefig('data_pic/correct_rate_by_label.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. 识别效率的分布
df['Efficiency'] = df['IsCorrect'] / df['ProcessingTime']
efficiency_by_label = df.groupby('True_Label')['Efficiency'].mean().reset_index()
efficiency_by_label['LabelName'] = label_names

# 打印数据到控制台
console.print("[bold magenta]不同标签的识别效率[/bold magenta]")
table = Table(show_header=True, header_style="bold blue")
table.add_column("标签", style="dim")
table.add_column("识别效率（正确率/处理时间）", justify="right")
for _, row in efficiency_by_label.iterrows():
    table.add_row(row['LabelName'], f"{row['Efficiency']:.2f}")
console.print(table)

# 绘制图表
plt.figure(figsize=(12, 6))
bars = sns.barplot(x='LabelName', y='Efficiency', hue='LabelName', data=efficiency_by_label, palette='plasma', legend=False)
plt.title('不同标签的识别效率', fontsize=16)
plt.xlabel('标签', fontsize=14)
plt.ylabel('识别效率（正确率/处理时间）', fontsize=14)
plt.xticks(rotation=45)
for bar in bars.patches:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom', fontsize=10)
plt.savefig('data_pic/efficiency_by_label.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. 按类别分析正确率
label_accuracy = df.groupby('Label_Name')['IsCorrect'].mean().sort_values()

# 打印数据到控制台
console.print("[bold magenta]各类别识别成功率[/bold magenta]")
table = Table(show_header=True, header_style="bold blue")
table.add_column("标签", style="dim")
table.add_column("成功率", justify="right")
for label, accuracy in label_accuracy.items():
    table.add_row(label, f"{accuracy:.2%}")
console.print(table)

# 绘制图表
plt.figure(figsize=(12, 6))
ax = label_accuracy.plot(kind='bar', color='skyblue', title='各类别识别成功率')
plt.ylabel('成功率')
plt.xticks(rotation=45)
for p in ax.patches:
    ax.annotate(f"{p.get_height():.2%}",
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 9),
                textcoords='offset points')
plt.savefig('data_pic/success_rate_by_label.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. 识别结果分布可视化
plt.figure(figsize=(10, 6))  # 调整整体图片宽度

# 各类别正确/失败数量
plt.subplot(1, 2, 2)
ax = sns.countplot(x='Label_Name', hue='IsCorrect', data=df)
plt.title('各类别识别结果分布', pad=10)  # 增加标题与图的间距
plt.xticks(rotation=45)
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}",
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 9),
                textcoords='offset points')

plt.tight_layout(pad=1.0)  # 减少子图之间的间距
plt.savefig('data_pic/result_distribution.png', dpi=300, bbox_inches='tight', pad_inches=0.05)  # 减少保存时的留白
plt.show()
# 9. 各类别处理时间的均值和标准差可视化
label_stats = df.groupby('Label_Name')['ProcessingTime'].agg(['mean', 'std']).reset_index()

# 打印数据到控制台
console.print("[bold magenta]各类别处理时间的均值和标准差[/bold magenta]")
table = Table(show_header=True, header_style="bold blue")
table.add_column("标签", style="dim")
table.add_column("平均处理时间（秒）", justify="right")
table.add_column("标准差", justify="right")
for _, row in label_stats.iterrows():
    table.add_row(row['Label_Name'], f"{row['mean']:.2f}", f"{row['std']:.2f}")
console.print(table)

# 绘制合并的柱状图
plt.figure(figsize=(14, 6))
x = range(len(label_stats))
width = 0.35

# 绘制均值柱状图
bars1 = plt.bar([i - width/2 for i in x], label_stats['mean'], width, color='skyblue', label='平均处理时间')
# 绘制标准差柱状图
bars2 = plt.bar([i + width/2 for i in x], label_stats['std'], width, color='lightcoral', label='标准差')

plt.title('各类别处理时间的均值和标准差', fontsize=16)
plt.xlabel('标签', fontsize=14)
plt.ylabel('时间（秒）', fontsize=14)
plt.xticks(x, label_stats['Label_Name'], rotation=45)
plt.legend()

# 添加数据标签
for bar in bars1:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.2f}', ha='center', va='bottom', fontsize=10)
for bar in bars2:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('data_pic/processing_time_stats_by_label.png', dpi=300, bbox_inches='tight')
plt.show()

# 10. 识别失败的标签类型分布
failed_df = df[df['IsCorrect'] == False]  # 筛选出识别失败的样本
failed_label_counts = failed_df['True_Label'].value_counts().sort_index()

# 打印数据到控制台
console.print("[bold magenta]识别失败的标签类型分布[/bold magenta]")
table = Table(show_header=True, header_style="bold blue")
table.add_column("True_Label", style="dim")
table.add_column("Label_Name", style="dim")
table.add_column("数量", justify="right")
for label, count in failed_label_counts.items():
    table.add_row(str(label), label_names[label], str(count))
console.print(table)

# 绘制图表
plt.figure(figsize=(12, 6))
x = np.arange(len(failed_label_counts))
bars = plt.bar(x, failed_label_counts.values, color='salmon')

# 设置x轴标签为True_Label (Label_Name)
plt.xticks(x, [f"{label} ({label_names[label]})" for label in failed_label_counts.index], rotation=45)

plt.title('识别失败的标签类型分布', fontsize=16)
plt.xlabel('True_Label (Label_Name)', fontsize=14)
plt.ylabel('数量', fontsize=14)

# 添加数据标签
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, int(height), ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig('data_pic/failed_label_distribution.png', dpi=300, bbox_inches='tight')
plt.show()