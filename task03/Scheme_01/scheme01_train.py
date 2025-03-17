import paddle
import pgl
import numpy as np
import pickle
import os
import pandas as pd  # 新增导入 pandas


class EnhancedGCN(paddle.nn.Layer):
    def __init__(self, input_size, num_class, hidden_size=64):
        super().__init__()
        self.gcn1 = pgl.nn.GCNConv(input_size, hidden_size, activation='relu')
        self.gcn2 = pgl.nn.GCNConv(hidden_size, hidden_size, activation='relu')
        self.dropout = paddle.nn.Dropout(p=0.3)
        self.classifier = paddle.nn.Linear(hidden_size, num_class)

        # 残差连接
        self.residual = paddle.nn.Linear(input_size, hidden_size) if input_size != hidden_size else None

    def forward(self, graph, feature):
        feature = paddle.to_tensor(feature, dtype='float32')  # 将 feature 转换为 Tensor
        residual = self.residual(feature) if self.residual else feature
        x = self.gcn1(graph, feature)
        x = x + residual  # 残差连接
        x = self.dropout(x)
        x = self.gcn2(graph, x)
        return self.classifier(x)


def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')


def load_data():
    data = np.load('graph_data_optimized.npz')
    node_feat = data['node_feat']
    edges = data['edges']
    edge_weights = data['edge_weights']

    # 标签严格对齐
    dataset_path = "c:/Users/13525/Desktop/cifar-10-batches-py/data_batch_1"
    batch = unpickle(dataset_path)
    labels = batch[b'labels'][:node_feat.shape[0]]  # 精确对齐

    # 打印调试信息
    print(f"原始标签: {labels}")
    print(f"原始标签类型: {type(labels)}")
    print(f"标签分布: {np.bincount(labels)}")  # 检查标签分布
    print(f"节点特征形状: {node_feat.shape}")
    print(f"边数量: {edges.shape[0]}")

    # 检查边的索引是否有效
    num_nodes = node_feat.shape[0]
    print(f"节点数量: {num_nodes}")  # 打印节点数量
    for edge in edges:
        if edge[0] < 0 or edge[1] < 0 or edge[0] >= num_nodes or edge[1] >= num_nodes:
            raise ValueError(f"无效的边索引: {edge}, 节点数量为 {num_nodes}")

    # 构建图结构
    graph = pgl.Graph(
        num_nodes=num_nodes,
        edges=edges,
        node_feat={'feat': node_feat},
        edge_feat={'weight': edge_weights.reshape(-1, 1)}
    )
    return graph, labels  # 返回原始标签


def train():
    """训练主函数"""
    print("初始化训练组件...")
    graph, labels = load_data()

    # 检查图结构是否有效
    if graph.num_nodes == 0:
        raise ValueError("图结构中节点数量为 0，请检查数据加载过程。")
    if graph.num_edges == 0:
        raise ValueError("图结构中边数量为 0，请检查数据加载过程。")

    # 将 labels 转换为 paddle.Tensor
    # 在 train() 函数中
    print(f"转换前标签: {labels}")
    print(f"转换前标签类型: {type(labels)}")

    # 手动创建张量并赋值
    labels_tensor = paddle.zeros([len(labels)], dtype='int64')  # 将 len(labels) 包装成 list
    for i, label in enumerate(labels):
        labels_tensor[i] = label
    labels = labels_tensor

    print(f"转换后标签: {labels.numpy()}")

    print(f"模型输入特征维度: {graph.node_feat['feat'].shape[1]}")

    # 手动保留节点数量
    num_nodes = graph.num_nodes

    # 将图转换为张量
    graph = graph.tensor()
    # 重新设置节点数量
    graph._num_nodes = num_nodes
    print("图已转换为张量")

    # 打印调试信息
    print(f"图节点数量: {graph.num_nodes}")
    print(f"图边数量: {graph.num_edges}")
    print(f"图节点特征维度: {graph.node_feat['feat'].shape}")

    # 初始化模型
    model = EnhancedGCN(
        input_size=graph.node_feat['feat'].shape[1],
        num_class=10
    )
    print("模型初始化完成")

    # 优化器配置
    optim = paddle.optimizer.Adam(
        learning_rate=0.0005,
        weight_decay=0.001,
        parameters=model.parameters()
    )
    criterion = paddle.nn.CrossEntropyLoss()

    # 训练循环
    best_loss = float('inf')
    patience, counter = 20, 0
    loss_history = []

    model.train()
    for epoch in range(300):
        logits = model(graph, graph.node_feat['feat'])
        loss = criterion(logits, labels)

        loss.backward()
        optim.step()
        optim.clear_grad()

        # 早停机制
        loss_value = loss.numpy()[0]
        loss_history.append(loss_value)
        if loss_value < best_loss:
            best_loss = loss_value
            counter = 0
            paddle.save(model.state_dict(), 'best_model.pdparams')
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        # 损失监控
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss_value:.4f}")

    # 分类验证
    model.set_state_dict(paddle.load('best_model.pdparams'))
    logits = model(graph, graph.node_feat['feat'])
    preds = paddle.argmax(logits, axis=1).numpy()

    # 打印调试信息
    print(f"预测结果: {preds}")
    print(f"实际标签: {labels.numpy()}")
    print(f"预测结果中的唯一值: {np.unique(preds)}")
    print(f"实际标签中的唯一值: {np.unique(labels.numpy())}")

    # 打印分类报告并保存为 CSV 文件
    from sklearn.metrics import classification_report
    report = classification_report(
        labels.numpy(), preds,
        labels=np.arange(10),  # 显式指定类别范围
        target_names=["飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"],
        output_dict=True,  # 输出为字典格式
        zero_division=1
    )
    # 将分类报告转换为 DataFrame 并保存为 CSV
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv('classification_report.csv', index=True)
    print("分类报告已保存为 classification_report.csv")

    # 打印分类报告到控制台
    print(classification_report(
        labels.numpy(), preds,
        labels=np.arange(10),
        target_names=["飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"],
        zero_division=1
    ))


if __name__ == '__main__':
    train()