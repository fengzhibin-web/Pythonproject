import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import classification_report
from visualdl import LogWriter
from sklearn.metrics import accuracy_score, top_k_accuracy_score

# 配置参数
config = {
    "hidden_dim": 512,
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 64,
    "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}


class ImprovedImageGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=102):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)  # 增加一层卷积
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = torch.relu(x)
        x = self.conv2(x, data.edge_index)
        x = torch.relu(x)
        x = self.conv3(x, data.edge_index)  # 新增卷积层
        graph_feature = global_mean_pool(x, data.batch)
        return self.classifier(graph_feature)


def load_and_split_data(npz_path):
    """加载数据"""
    with np.load(npz_path) as data:
        return {
            'edges': data['edges'],
            'node_feat': data['node_feat'],
            'labels': data['labels'],
            'node_image_map': data['node_image_map']
        }


def create_pyg_data(npz_data):
    """创建PyG图数据"""
    edges = npz_data['edges']
    # 提取边的节点索引，忽略权重
    edge_index = torch.tensor(edges[:, :2].T, dtype=torch.long)
    x = torch.tensor(npz_data['node_feat'], dtype=torch.float)
    y = torch.tensor(npz_data['labels'], dtype=torch.long)
    batch = torch.tensor(npz_data['node_image_map'], dtype=torch.long)

    # 添加维度检查
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index 维度错误，应为 (2, num_edges)，实际为 {edge_index.shape}")
    if x.dim() != 2:
        raise ValueError(f"节点特征维度错误，应为 (num_nodes, num_features)，实际为 {x.shape}")
    if y.dim() != 1:
        raise ValueError(f"标签维度错误，应为 (num_nodes,)，实际为 {y.shape}")
    if batch.dim() != 1:
        raise ValueError(f"批次维度错误，应为 (num_nodes,)，实际为 {batch.shape}")

    return Data(x=x, edge_index=edge_index, y=y, batch=batch)


# 初始化 VisualDL
log_writer = LogWriter(logdir="./log")


def main():
    try:
        # 加载测试数据
        print("正在加载训练测试数据...")
        train_data = load_and_split_data("D:/Pythonproject/task04/train_graph_test.npz")
        print("正在加载验证测试数据...")
        val_data = load_and_split_data("D:/Pythonproject/task04/val_graph_test.npz")

        # 创建图数据
        print("正在创建训练图数据...")
        train_pyg = create_pyg_data(train_data).to(config['device'])
        print("正在创建验证图数据...")
        val_pyg = create_pyg_data(val_data).to(config['device'])

        # 初始化模型
        model = ImprovedImageGCN(input_dim=512, hidden_dim=config['hidden_dim']).to(config['device'])
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0  # 记录最佳验证集准确率
        best_model_path = "final_model.pth"  # 修改模型保存路径

        # 训练循环
        for epoch in range(config['epochs']):
            model.train()
            optimizer.zero_grad()
            output = model(train_pyg)
            loss = criterion(output, train_pyg.y)
            loss.backward()
            optimizer.step()

            # 计算训练集准确率
            pred = output.argmax(dim=1).cpu().numpy()
            true_labels = train_pyg.y.cpu().numpy()
            train_acc = accuracy_score(true_labels, pred)

            # 记录训练指标
            log_writer.add_scalar(tag="Loss", step=epoch, value=loss.item())
            log_writer.add_scalar(tag="Train_Accuracy", step=epoch, value=train_acc)

            # 每 5 个 epoch 打印一次损失和准确率
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{config['epochs']}, Loss: {loss.item():.4f}, Train Accuracy: {train_acc:.4f}")

        # 训练结束后保存最后一个模型
        torch.save(model.state_dict(), best_model_path)
        print(f"训练完成，最终模型已保存到 {best_model_path}")

        # 测试
        model.eval()
        with torch.no_grad():
            val_output = model(val_pyg)
            pred = val_output.argmax(dim=1).cpu().numpy()
            true_labels = val_pyg.y.cpu().numpy()
            val_acc = accuracy_score(true_labels, pred)

            # 如果当前验证集准确率更高，保存模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
                print(f"发现更好的模型，验证集准确率: {val_acc:.4f}，模型已保存到 {best_model_path}")

        # 记录测试指标
        log_writer.add_scalar(tag="Val_Accuracy", step=config['epochs'], value=val_acc)

        # 输出分类报告
        print("\n分类报告:")
        print(classification_report(true_labels, pred, digits=4))

    except Exception as e:
        print(f"程序运行出错: {str(e)}")


if __name__ == "__main__":
    main()