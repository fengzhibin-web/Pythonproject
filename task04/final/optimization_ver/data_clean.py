import os
import cv2
import numpy as np
import matplotlib.pyplot as plt  # 使用 matplotlib 库
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def check_images(image_paths):
    """检查图像文件是否有效"""
    valid_images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            valid_images.append(path)
        else:
            print(f"警告: 无法读取图像 {path}，跳过该图像")
    return valid_images

def check_labels(labels):
    """检查标签是否有效"""
    valid_labels = []
    for label in labels:
        if 0 <= label < 102:  # 标签范围是 0 到 101
            valid_labels.append(label)
        else:
            print(f"警告: 无效标签 {label}，跳过该标签")
    return valid_labels

def remove_duplicates(image_paths, labels):
    """去除重复的图像和标签"""
    unique_data = {}
    for path, label in zip(image_paths, labels):
        if path not in unique_data:
            unique_data[path] = label
        else:
            print(f"警告: 发现重复图像 {path}，跳过该图像")
    return list(unique_data.keys()), list(unique_data.values())

def convert_image_format(image_paths, target_format=".jpg"):
    """统一图像格式"""
    converted_images = []
    for path in image_paths:
        if not path.lower().endswith(target_format):
            new_path = os.path.splitext(path)[0] + target_format
            img = cv2.imread(path)
            cv2.imwrite(new_path, img)
            converted_images.append(new_path)
            print(f"已将图像 {path} 转换为 {new_path}")
        else:
            converted_images.append(path)
    return converted_images

def clean_data(image_paths, labels):
    """数据清洗主函数"""
    print("正在清洗数据...")
    image_paths = check_images(image_paths)
    labels = check_labels(labels)
    image_paths, labels = remove_duplicates(image_paths, labels)
    image_paths = convert_image_format(image_paths)
    
    # 检查是否有异常
    if len(image_paths) == 0 or len(labels) == 0:
        print("数据清洗完成，发现异常")
    else:
        print("数据清洗完成，无异常")
    return image_paths, labels

def plot_and_save_label_distribution(labels, title, filename):
    """绘制并保存标签分布图"""
    unique_labels, counts = np.unique(labels, return_counts=True)
    plt.figure(figsize=(12, 6))
    plt.bar(unique_labels, counts)
    plt.xlabel('类别')
    plt.ylabel('数量')
    plt.title(title)
    plt.savefig(filename)  # 保存图片
    plt.close()

def plot_dataset_summary(train_image_count, train_label_count, val_image_count, val_label_count):
    """绘制数据集摘要图"""
    datasets = ['训练集', '验证集']
    image_counts = [train_image_count, val_image_count]
    label_counts = [train_label_count, val_label_count]

    x = np.arange(len(datasets))  # 横轴位置
    width = 0.35  # 柱子宽度

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, image_counts, width, label='图片总数')
    rects2 = ax.bar(x + width/2, label_counts, width, label='标签总数')

    # 添加标签和标题
    ax.set_xlabel('数据集')
    ax.set_ylabel('数量')
    ax.set_title('数据集摘要')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()

    # 在柱子上方显示数量
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    plt.tight_layout()
    plt.savefig("dataset_summary.png")  # 保存图片
    plt.close()

def main():
    # 加载 flowers102 数据集
    with open("dataset/flowers102/train_list.txt", "r") as f:
        train_lines = f.readlines()  # 读取所有训练集图片
    with open("dataset/flowers102/val_list.txt", "r") as f:
        val_lines = f.readlines()  # 读取所有验证集图片

    # 提取训练集和验证集图像路径及标签
    train_image_paths = [f"dataset/flowers102/{line.split()[0]}" for line in train_lines]
    train_labels = [int(line.split()[1]) for line in train_lines]
    val_image_paths = [f"dataset/flowers102/{line.split()[0]}" for line in val_lines]
    val_labels = [int(line.split()[1]) for line in val_lines]

    # 计算图片总数和标签总数
    train_image_count = len(train_image_paths)
    train_label_count = len(np.unique(train_labels))
    val_image_count = len(val_image_paths)
    val_label_count = len(np.unique(val_labels))

    # 绘制数据集摘要图
    plot_dataset_summary(train_image_count, train_label_count, val_image_count, val_label_count)

    # 数据清洗
    train_image_paths, train_labels = clean_data(train_image_paths, train_labels)
    val_image_paths, val_labels = clean_data(val_image_paths, val_labels)

    print(f"清洗后训练集图像数量: {len(train_image_paths)}")
    print(f"清洗后验证集图像数量: {len(val_image_paths)}")

if __name__ == "__main__":
    main()
