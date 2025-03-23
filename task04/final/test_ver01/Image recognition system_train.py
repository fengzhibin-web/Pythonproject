import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import classification_report

# 配置参数
config = {
    "hidden_dim": 512,
    "learning_rate": 0.0005,
    "epochs": 200,
    "batch_size": 32,
    "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

class ImageGCN(nn.Module):
    """GCN模型结构"""
    def __init__(self, input_dim, hidden_dim, output_dim=102):  # 102 个类别
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
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
        graph_feature = global_mean_pool(x, data.batch)
        return self.classifier(graph_feature)

def load_and_split_data(npz_path):
    """加载数据并划分数据集"""
    with np.load(npz_path) as data:
        return {
            'edges': data['edges'],
            'node_feat': data['node_feat'],
            'labels': data['labels'],
            'node_image_map': data['node_image_map']
        }

def create_pyg_data(npz_data):
    """创建PyG图数据"""
    edge_index = torch.tensor(npz_data['edges'].T, dtype=torch.long)
    x = torch.tensor(npz_data['node_feat'], dtype=torch.float)
    y = torch.tensor(npz_data['labels'], dtype=torch.long)
    batch = torch.tensor(npz_data['node_image_map'], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y, batch=batch)

def main():
    # 加载数据
    train_data = load_and_split_data("../../train_graph.npz")
    val_data = load_and_split_data("../../val_graph.npz")
    train_pyg = create_pyg_data(train_data).to(config['device'])
    val_pyg = create_pyg_data(val_data).to(config['device'])

    # 初始化模型
    model = ImageGCN(input_dim=512, hidden_dim=config['hidden_dim']).to(config['device'])
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    for epoch in range(config['epochs']):
        model.train()
        optimizer.zero_grad()
        output = model(train_pyg)
        loss = criterion(output, train_pyg.y)
        loss.backward()
        optimizer.step()

        # 每 10 个 epoch 打印一次损失
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{config['epochs']}, Loss: {loss.item():.4f}")

    # 测试
    model.eval()
    with torch.no_grad():
        val_output = model(val_pyg)
        pred = val_output.argmax(dim=1).cpu().numpy()
        true_labels = val_pyg.y.cpu().numpy()

    # 输出分类报告
    print("\n分类报告:")
    print(classification_report(true_labels, pred, digits=4))

if __name__ == "__main__":
    main()