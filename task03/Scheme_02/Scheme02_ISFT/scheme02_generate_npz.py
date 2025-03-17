import numpy as np
import pickle
import cv2
import pgl
from pgl.graph import Graph


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def extract_sift_features(image):
    """提取SIFT特征"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    print(f"检测到关键点数量: {len(kp) if kp else 0}")
    return des if des is not None else np.zeros((1, 128))


def build_spatial_edges(features, image_size=32, grid_size=8):
    """构建基于空间关系的边"""
    print(f"\n构建空间关系边，特征数量: {len(features)}")
    edges = []
    cell_size = image_size // grid_size

    # 将特征点分配到网格
    grid = {}
    for idx, (x, y) in enumerate(features[:, :2]):  # 使用特征坐标前两维作为位置
        i, j = int(x // cell_size), int(y // cell_size)
        grid.setdefault((i, j), []).append(idx)

    # 为相邻网格中的特征点创建边
    for (i, j), points in grid.items():
        # 连接同一单元格内的特征点
        for p1 in points:
            for p2 in points:
                if p1 != p2:
                    edges.append((p1, p2))

        # 连接相邻单元格的特征点
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                ni, nj = i + di, j + dj
                if (ni, nj) in grid:
                    for p1 in points:
                        for p2 in grid[(ni, nj)]:
                            edges.append((p1, p2))
    print(f"生成的边数量: {len(edges)}")
    return edges


def extract_features(images):
    """提取图像特征"""
    all_features = []
    node_image_map = []
    for img_idx, image in enumerate(images):
        features = extract_sift_features(image)
        spatial_info = np.array([[x, y] for y in range(32) for x in range(32)])
        if len(features) > 0:
            features = np.hstack([features, spatial_info[:len(features)]])
            all_features.extend(features)
            node_image_map.extend([img_idx] * len(features))
        print(f"图像 {img_idx} 提取到特征数量: {len(features)}")
    return np.array(all_features, dtype=np.float32), np.array(node_image_map)

def main():
    # 加载CIFAR-10数据集
    print("\n加载CIFAR-10数据集")
    dataset_path = "c:/Users/13525/Desktop/cifar-10-batches-py/data_batch_1"
    batch = unpickle(dataset_path)
    data = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    labels = batch[b'labels']
    print(f"加载数据数量: {len(data)}")
    # 提取前500张图片的特征
    train_data = data[:500]
    train_labels = labels[:500]
    train_features, train_node_image_map = extract_features(train_data)

    # 提取第501-1000张图片的特征
    test_data = data[500:1000]
    test_labels = labels[500:1000]
    test_features, test_node_image_map = extract_features(test_data)

    # 转换为numpy数组
    train_node_feat = np.array(train_features, dtype=np.float32)
    test_node_feat = np.array(test_features, dtype=np.float32)

    # 构建空间关系边
    train_edges = build_spatial_edges(train_node_feat)
    test_edges = build_spatial_edges(test_node_feat)

    # 生成边数组
    train_edge_src = np.array([e[0] for e in train_edges], dtype=np.int64)
    train_edge_dst = np.array([e[1] for e in train_edges], dtype=np.int64)
    test_edge_src = np.array([e[0] for e in test_edges], dtype=np.int64)
    test_edge_dst = np.array([e[1] for e in test_edges], dtype=np.int64)

    # 创建PGL图结构
    train_graph = Graph(
        num_nodes=len(train_node_feat),
        edges=np.column_stack([train_edge_src, train_edge_dst]),
        node_feat={'feat': train_node_feat}
    )
    test_graph = Graph(
        num_nodes=len(test_node_feat),
        edges=np.column_stack([test_edge_src, test_edge_dst]),
        node_feat={'feat': test_node_feat}
    )

    # 保存训练数据
    save_train_path = 'train_sift_graph.npz'
    np.savez(save_train_path,
             node_feat=train_node_feat,
             edges=np.column_stack([train_edge_src, train_edge_dst]),
             node_image_map=train_node_image_map,
             labels=np.array(train_labels))

    # 保存测试数据
    save_test_path = 'test_sift_graph.npz'
    np.savez(save_test_path,
             node_feat=test_node_feat,
             edges=np.column_stack([test_edge_src, test_edge_dst]),
             node_image_map=test_node_image_map,
             labels=np.array(test_labels))

    # 训练数据保存验证
    print("\n训练文件保存验证:")
    import os
    abs_train_path = os.path.abspath(save_train_path)
    print(f"保存前特征检查 - 最小值: {np.min(train_node_feat)}, 最大值: {np.max(train_node_feat)}")
    print(f"文件路径: {abs_train_path}")
    print(f"文件大小: {os.path.getsize(abs_train_path) / 1024:.1f}KB")

    try:
        with np.load(save_train_path) as data:
            print("\n加载文件内容验证:")
            print(f"加载后特征检查 - 最小值: {np.min(data['node_feat'])}, 最大值: {np.max(data['node_feat'])}")
            print(f"节点特征: {data['node_feat'].shape} {data['node_feat'].dtype}")
            print(f"边数据: {data['edges'].shape} {data['edges'].dtype}")
            print(f"标签数量: {len(data['labels'])}")
            print(f"节点-图像映射有效性: {np.unique(data['node_image_map']).shape[0]}张图像的映射")
    except Exception as e:
        print(f"\n加载失败: {str(e)}")

    # 测试数据保存验证
    print("\n测试文件保存验证:")
    abs_test_path = os.path.abspath(save_test_path)
    print(f"保存前特征检查 - 最小值: {np.min(train_node_feat)}, 最大值: {np.max(train_node_feat)}")
    print(f"文件路径: {abs_test_path}")
    print(f"文件大小: {os.path.getsize(abs_test_path) / 1024:.1f}KB")

    try:
        with np.load(save_test_path) as data:
            print("\n加载文件内容验证:")
            print(f"加载后特征检查 - 最小值: {np.min(data['node_feat'])}, 最大值: {np.max(data['node_feat'])}")
            print(f"节点特征: {data['node_feat'].shape} {data['node_feat'].dtype}")
            print(f"边数据: {data['edges'].shape} {data['edges'].dtype}")
            print(f"标签数量: {len(data['labels'])}")
            print(f"节点-图像映射有效性: {np.unique(data['node_image_map']).shape[0]}张图像的映射")
    except Exception as e:
        print(f"\n加载失败: {str(e)}")

    print(f"训练数据已保存至: {save_train_path}")
    print(f"测试数据已保存至: {save_test_path}")


if __name__ == '__main__':
    main()