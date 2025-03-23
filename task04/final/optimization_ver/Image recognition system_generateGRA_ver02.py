import numpy as np
import pickle
import cv2
from paddlex import create_model
from scipy.spatial import KDTree
import pgl
from pgl.graph import Graph

# 初始化 PP-ShiTuV2 模型
model = create_model("PP-ShiTuV2_rec")


def extract_features_with_ppshitu(image_paths):
    """使用预训练模型PP-ShiTuV2 提取图像特征"""
    features = []
    for idx, path in enumerate(image_paths):
        print(f"正在处理第 {idx + 1}/{len(image_paths)} 张图像: {path}")
        img = cv2.imread(path)
        if img is None:
            print(f"警告: 无法读取图像 {path}，跳过该图像")
            continue
        output = list(model.predict(img, batch_size=1))
        feature = output[0]['rec_feature']  # 提取 512 维特征
        # 修改特征归一化方式
        feature = feature / (np.linalg.norm(feature) + 1e-8)  # L2 归一化
        features.append(feature)
        print(f"特征提取完成，特征维度: {feature.shape}")
    return np.array(features)


def build_spatial_edges(features, max_neighbors=10):
    # 计算特征距离的均值作为动态阈值
    distances = np.linalg.norm(features[:, np.newaxis] - features, axis=2)
    distance_threshold = np.mean(distances) * 0.8  # 动态调整阈值
    edges = []
    kdtree = KDTree(features)
    print(f"特征矩阵形状: {features.shape}")
    print(f"特征示例值: {features[0][:5]}")  # 打印前5个特征值用于调试

    for i, feature in enumerate(features):
        print(f"\n正在为第 {i + 1}/{len(features)} 个节点构建边")
        # 使用 KNN 和距离阈值筛选边
        dists, indices = kdtree.query(feature, k=max_neighbors + 1)
        print(f"最近邻距离: {dists[1:]}")  # 打印最近邻距离用于调试

        valid_edges = 0
        for j, dist in zip(indices[1:], dists[1:]):  # 排除自身
            if dist < distance_threshold:
                # 结合特征相似度与空间距离计算边权重
                feature_sim = np.dot(feature, features[j]) / (np.linalg.norm(feature) * np.linalg.norm(features[j]))
                edge_weight = (1 - dist) * feature_sim
                edges.append([i, j, edge_weight])
                valid_edges += 1
                print(f"找到有效边: {i} -> {j}, 距离: {dist:.4f}, 权重: {edge_weight:.4f}")

        print(f"为节点 {i} 找到 {valid_edges} 个有效邻居")

    if len(edges) == 0:
        print("警告：没有找到任何有效的边，将创建默认边")
        # 创建默认的全连接图
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                edges.append([i, j, 1.0])

    return np.array(edges)


def save_graph_data(features, edges, labels, node_image_map, output_path):
    """保存图数据为 .npz 文件"""
    # 调整边数据格式为 (2, num_edges)
    edges = np.array(edges)[:, :2].T  # 只保留 src 和 dst，并转置为 (2, num_edges)

    # 将 labels 转换为 NumPy 数组
    labels = np.array(labels)

    # 打印调试信息
    print("\n保存数据前调试信息：")
    print(f"节点特征形状: {features.shape}")
    print(f"边数据形状: {edges.shape}")
    print(f"标签形状: {labels.shape}")
    print(f"节点图像映射形状: {node_image_map.shape}")

    np.savez_compressed(
        output_path,
        node_feat=features,
        edges=edges,
        labels=labels,
        node_image_map=node_image_map
    )


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

    # 生成边数组
    train_edge_src = np.array([e[0] for e in train_edges], dtype=np.int64)
    train_edge_dst = np.array([e[1] for e in train_edges], dtype=np.int64)
    val_edge_src = np.array([e[0] for e in val_edges], dtype=np.int64)
    val_edge_dst = np.array([e[1] for e in val_edges], dtype=np.int64)

    # 创建PGL图结构
    train_graph = Graph(
        num_nodes=len(train_features),
        edges=np.column_stack([train_edge_src, train_edge_dst]),
        node_feat={'feat': train_features}
    )
    val_graph = Graph(
        num_nodes=len(val_features),
        edges=np.column_stack([val_edge_src, val_edge_dst]),
        node_feat={'feat': val_features}
    )

    # 保存图数据
    train_node_image_map = np.arange(len(train_features))
    val_node_image_map = np.arange(len(val_features))
    save_graph_data(train_features, train_edges, train_labels, train_node_image_map, "train_graph_test.npz")
    save_graph_data(val_features, val_edges, val_labels, val_node_image_map, "val_graph_test.npz")
    print("测试图数据保存完成")


if __name__ == "__main__":
    main()