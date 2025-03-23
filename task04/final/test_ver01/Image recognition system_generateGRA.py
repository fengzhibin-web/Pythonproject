import numpy as np
import pickle
import cv2
from paddlex import create_model
from scipy.spatial import KDTree

# 初始化 PP-ShiTuV2 模型
model = create_model("PP-ShiTuV2_rec")

def extract_features_with_ppshitu(image_paths):
    """使用 PP-ShiTuV2 提取图像特征"""
    features = []
    for idx, path in enumerate(image_paths):
        print(f"正在处理第 {idx + 1}/{len(image_paths)} 张图像: {path}")
        img = cv2.imread(path)
        if img is None:
            print(f"警告: 无法读取图像 {path}，跳过该图像")
            continue
        output = list(model.predict(img, batch_size=1))
        feature = output[0]['rec_feature']  # 提取 512 维特征
        features.append(feature)
        print(f"特征提取完成，特征维度: {feature.shape}")
    return np.array(features)

def build_spatial_edges(features, max_neighbors=10):
    """构建基于空间关系的边"""
    edges = []
    kdtree = KDTree(features)
    for i, feature in enumerate(features):
        print(f"正在为第 {i + 1}/{len(features)} 个节点构建边")
        _, indices = kdtree.query(feature, k=max_neighbors + 1)
        for j in indices[1:]:  # 排除自身
            edges.append([i, j])
        print(f"为节点 {i} 找到 {len(indices) - 1} 个邻居")
    return np.array(edges)

def save_graph_data(features, edges, labels, node_image_map, output_path):
    """保存图数据为 .npz 文件"""
    np.savez(
        output_path,
        node_feat=features,
        edges=edges,
        labels=labels,
        node_image_map=node_image_map
    )

def main():
    # 加载 flowers102 数据集
    with open("dataset/flowers102/train_list.txt", "r") as f:
        train_lines = f.readlines()
    with open("dataset/flowers102/val_list.txt", "r") as f:
        val_lines = f.readlines()

    # 提取训练集和验证集图像路径及标签
    train_image_paths = [f"dataset/flowers102/{line.split()[0]}" for line in train_lines]
    train_labels = [int(line.split()[1]) for line in train_lines]
    val_image_paths = [f"dataset/flowers102/{line.split()[0]}" for line in val_lines]
    val_labels = [int(line.split()[1]) for line in val_lines]

    print(f"训练集图像数量: {len(train_image_paths)}")
    print(f"验证集图像数量: {len(val_image_paths)}")

    # 提取特征
    print("正在提取训练集特征...")
    train_features = extract_features_with_ppshitu(train_image_paths)
    print("正在提取验证集特征...")
    val_features = extract_features_with_ppshitu(val_image_paths)

    # 构建图结构
    print("正在构建训练集图结构...")
    train_edges = build_spatial_edges(train_features)
    print("正在构建验证集图结构...")
    val_edges = build_spatial_edges(val_features)

    # 保存图数据
    train_node_image_map = np.arange(len(train_features))
    val_node_image_map = np.arange(len(val_features))
    save_graph_data(train_features, train_edges, train_labels, train_node_image_map, "../../train_graph.npz")
    save_graph_data(val_features, val_edges, val_labels, val_node_image_map, "../../val_graph.npz")
    print("图数据保存完成")

if __name__ == "__main__":
    main()