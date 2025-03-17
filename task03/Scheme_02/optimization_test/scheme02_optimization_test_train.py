import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from tqdm import tqdm
import time
from sklearn.metrics import classification_report

# 配置参数
config = {
    "hidden_dim": 512,  # 增加隐藏层维度
    "learning_rate": 0.0005,  # 降低学习率
    "epochs": 200,  # 增加训练轮数
    "batch_size": 32,  # 调整batch size
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}


class ImageGCN(nn.Module):
    """GCN模型结构"""

    def __init__(self, input_dim, hidden_dim, output_dim=10):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # 添加BatchNorm
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)  # 添加BatchNorm
        self.dropout = nn.Dropout(0.5)  # 添加Dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, data):
        # 图卷积
        x = self.conv1(data.x, data.edge_index)
        x = torch.relu(x)
        x = self.conv2(x, data.edge_index)

        # 全局池化 (图像级别)
        graph_feature = global_mean_pool(x, data.batch)
        return self.classifier(graph_feature)


def load_and_split_data(npz_path):
    """加载数据并划分数据集"""
    # 使用with语句确保文件及时关闭
    with np.load(npz_path) as data:
        # 按原始图像划分数据集
        unique_images = np.unique(data['node_image_map'])
        n_total = len(unique_images)

        # 随机划分
        np.random.shuffle(unique_images)
        train_images = unique_images[:int(n_total * config["train_ratio"])]
        val_images = unique_images[len(train_images):len(train_images) + int(n_total * config["val_ratio"])]
        test_images = unique_images[len(train_images) + len(val_images):]

        # 创建节点掩码
        train_mask = np.isin(data['node_image_map'], train_images)
        val_mask = np.isin(data['node_image_map'], val_images)
        test_mask = np.isin(data['node_image_map'], test_images)

        # 只返回必要的数据
        return {
            'edges': data['edges'],
            'node_feat': data['node_feat'],
            'labels': data['labels'],
            'node_image_map': data['node_image_map']
        }, (train_mask, val_mask, test_mask)

def create_pyg_data(npz_data, mask):
    """创建PyG图数据"""
    # 获取子图节点
    node_indices = np.where(mask)[0]
    subgraph_nodes = np.unique(np.concatenate([
        npz_data['edges'][:, 0],
        npz_data['edges'][:, 1]
    ]))

    # 构建图数据
    edge_index = torch.tensor(npz_data['edges'].T, dtype=torch.long)
    x = torch.tensor(npz_data['node_feat'], dtype=torch.float)
    x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)  # 添加归一化
    y = torch.tensor(npz_data['labels'], dtype=torch.long)

    # 添加batch信息
    batch = torch.tensor(npz_data['node_image_map'], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y, batch=batch)


if __name__ == "__main__":
    # 打印设备信息
    print(f"[INFO] 使用设备: {config['device']}")

    # 添加内存清理机制
    torch.cuda.empty_cache()

    # 加载训练数据
    print("[INFO] 正在加载训练数据...")
    start_time = time.time()
    train_npz_data = load_and_split_data("../train_sift_graph.npz")
    print(f"[DEBUG] 训练数据统计 - 节点数量: {train_npz_data[0]['node_feat'].shape[0]}, 边数量: {train_npz_data[0]['edges'].shape[0]}")
    train_data = create_pyg_data(train_npz_data[0], train_npz_data[1][0]).to(config['device'])
    del train_npz_data
    print(f"[INFO] 训练数据加载完成，耗时: {time.time() - start_time:.2f}秒")
    print(f"[DEBUG] 训练图结构 - 节点特征维度: {train_data.x.shape}, 边数量: {train_data.edge_index.shape[1]}")

    # 加载测试数据
    print("[INFO] 正在加载测试数据...")
    start_time = time.time()
    test_npz_data = load_and_split_data("../test_sift_graph.npz")
    print(f"[DEBUG] 测试数据统计 - 节点数量: {test_npz_data[0]['node_feat'].shape[0]}, 边数量: {test_npz_data[0]['edges'].shape[0]}")
    test_data = create_pyg_data(test_npz_data[0], test_npz_data[1][2]).to(config['device'])
    del test_npz_data
    print(f"[INFO] 测试数据加载完成，耗时: {time.time() - start_time:.2f}秒")
    print(f"[DEBUG] 测试图结构 - 节点特征维度: {test_data.x.shape}, 边数量: {test_data.edge_index.shape[1]}")

    # 初始化模型
    print("[INFO] 正在初始化模型...")
    model = ImageGCN(
        input_dim=train_data.x.shape[1],
        hidden_dim=config["hidden_dim"]
    ).to(config['device'])
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # 每30个epoch学习率衰减
    class_counts = torch.bincount(train_data.y)
    class_weights = 1. / class_counts.float()
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(config['device']))
    print("[INFO] 模型初始化完成")
    print(f"[DEBUG] 模型参数数量: {sum(p.numel() for p in model.parameters())}")

    # 训练循环
    print(f"[INFO] 开始训练，共 {config['epochs']} 个epoch")
    best_loss = float('inf')
    total_train_time = 0.0  # 添加总训练时间统计
    
    # 开始计时
    train_start_time = time.time()
    
    for epoch in tqdm(range(config["epochs"]), desc="训练进度"):
        epoch_start_time = time.time()  # 记录epoch开始时间
        
        model.train()
        optimizer.zero_grad()
        output = model(train_data)
        loss = criterion(output, train_data.y)
        loss.backward()
        optimizer.step()

        # 更新最佳损失
        if loss.item() < best_loss:
            best_loss = loss.item()

        # 每10个epoch打印一次损失并清理内存
        if (epoch + 1) % 10 == 0:
            print(f"[INFO] Epoch {epoch + 1:03d} | Train Loss: {loss:.4f} | Best Loss: {best_loss:.4f}")
            torch.cuda.empty_cache()
            
        # 计算并累加epoch耗时
        epoch_time = time.time() - epoch_start_time
        total_train_time += epoch_time

    # 计算总耗时和平均耗时
    total_time = time.time() - train_start_time
    avg_epoch_time = total_train_time / config["epochs"]
    
    # 打印耗时统计
    print(f"\n[INFO] 训练完成")
    print(f"[INFO] 总训练耗时: {total_time:.2f}秒")
    print(f"[INFO] 平均每个epoch耗时: {avg_epoch_time:.2f}秒")

    
    # 定义类别名称
    class_names = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']
    
    # 修改测试部分代码
    print("[INFO] 开始测试...")
    model.eval()
    
    # 开始测试计时
    test_start_time = time.time()
    
    with torch.no_grad():
        test_output = model(test_data)
        pred = test_output.argmax(dim=1).cpu().numpy()
        true_labels = test_data.y.cpu().numpy()
        
        # 计算测试耗时
        test_time = time.time() - test_start_time
        avg_process_time = test_time / len(test_data.y)
        
        # 生成分类报告
        print("\n分类报告:")
        print(classification_report(true_labels, pred, target_names=class_names, digits=2))
        
        # 保留原有输出
        accuracy = (pred == true_labels).mean()
        print(f"\n[INFO] 测试准确率: {accuracy:.4f}")
        print(f"[DEBUG] 预测标签分布: {np.bincount(pred)}")
        print(f"[DEBUG] 真实标签分布: {np.bincount(true_labels)}")
        
        # 打印测试耗时统计
        print(f"\n[INFO] 测试总耗时: {test_time:.4f}秒")
        print(f"[INFO] 平均处理时间: {avg_process_time:.6f}秒/样本")