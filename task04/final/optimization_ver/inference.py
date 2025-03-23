import numpy as np
import torch
import time
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn

# 配置参数
config = {
    "hidden_dim": 512,
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

class ImprovedImageGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=102):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)  # 增加第三层卷积
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
        x = self.conv3(x, data.edge_index)  # 使用第三层卷积
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

def load_model(model_path):
    """加载训练好的模型"""
    model = ImprovedImageGCN(input_dim=512, hidden_dim=config['hidden_dim'])  # 使用 ImprovedImageGCN
    model.load_state_dict(torch.load(model_path))
    model.to(config['device'])
    model.eval()
    return model

def inference(model, data, num_samples=10):
    """推理函数"""
    # 随机抽取 num_samples 张图片
    indices = np.random.choice(len(data['labels']), num_samples, replace=False)
    sampled_data = {
        'edges': data['edges'],
        'node_feat': data['node_feat'][indices],
        'labels': data['labels'][indices],
        'node_image_map': data['node_image_map'][indices]
    }
    pyg_data = create_pyg_data(sampled_data).to(config['device'])

    # 推理并计算耗时
    start_time = time.time()
    with torch.no_grad():
        output = model(pyg_data)
    end_time = time.time()
    inference_time = end_time - start_time

    # 计算正确率
    pred = output.argmax(dim=1).cpu().numpy()
    true_labels = pyg_data.y.cpu().numpy()
    accuracy = np.mean(pred == true_labels)

    # 输出每张图片的真实标签和预测标签
    print("\n调试信息：")
    for i, (true_label, pred_label) in enumerate(zip(true_labels, pred)):
        print(f"图片 {i + 1}: 真实标签 = {true_label}, 预测标签 = {pred_label}")

    return accuracy, inference_time

def main():
    # 加载模型
    model = load_model("final_model.pth")

    # 加载数据集
    data = load_and_split_data("val_graph_test.npz")

    # 进行推理
    total_accuracy = 0.0
    total_time = 0.0
    num_trials = 10  # 重复 10 次推理
    for _ in range(num_trials):
        accuracy, inference_time = inference(model, data)
        total_accuracy += accuracy
        total_time += inference_time

    # 计算平均正确率和平均耗时
    avg_accuracy = total_accuracy / num_trials
    avg_time = total_time / num_trials

    print(f"平均正确率: {avg_accuracy:.4f}")
    print(f"平均推理耗时: {avg_time:.4f} 秒")

if __name__ == "__main__":
    main()
