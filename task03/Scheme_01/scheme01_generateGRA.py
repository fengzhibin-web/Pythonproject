# encoding:utf-8
import requests
import base64
import numpy as np
import pickle
from PIL import Image
import io
import pgl
import networkx as nx
from sklearn.metrics import jaccard_score


def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')


def get_access_token():
    API_KEY = "3hRiudPTxVVhq2l5N4AUui69"
    SECRET_KEY = "P654iKyrFuzs9Dv39QA7QAs5Ngb1o9yp"
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))


def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def main():
    # 配置参数
    NUM_IMAGES = 500
    MIN_CONFIDENCE = 0.6
    SIMILARITY_PERCENTILE = 60

    # 数据加载
    dataset_path = "c:/Users/13525/Desktop/cifar-10-batches-py/data_batch_1"
    batch = unpickle(dataset_path)
    data, labels = batch[b'data'], batch[b'labels']

    # 语义特征提取
    semantic_features = []  # 初始化 semantic_features 为空列表
    all_keywords = set()

    for i in range(NUM_IMAGES):
        # 获取图片数据并转换为RGB格式
        image_data = data[i].reshape(3, 32, 32).transpose(1, 2, 0)
        image_data = np.ascontiguousarray(image_data)

        if image_data.dtype != np.uint8:
            image_data = image_data.astype(np.uint8)

        # 图片预处理
        img = Image.fromarray(image_data, 'RGB').resize((64, 64))
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # API请求（增加重试机制）
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url="https://aip.baidubce.com/rest/2.0/image-classify/v2/advanced_general",
                    headers={'content-type': 'application/x-www-form-urlencoded'},
                    params={
                        "access_token": get_access_token(),
                        "image": image_base64,
                        "baike_num": 0
                    },
                    timeout=10
                )
                response.raise_for_status()
                result = response.json()  # 正确定义result变量
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"第{i + 1}张图片API请求失败: {str(e)}")
                    result = {'result': []}  # 确保result被定义
                    break

        # 修改关键词筛选策略
        if result.get('result'):
            valid_results = [r for r in result['result'] if r['score'] >= MIN_CONFIDENCE]
        top_results = valid_results[:5]  # 取前5个高置信度结果
        keywords = [f"{r['keyword']}_{r['root']}" for r in top_results]  # 增加语义粒度
        semantic_features.append(keywords)
        all_keywords.update(keywords)

    # 特征工程增强
    keyword_idx = {kw: i for i, kw in enumerate(all_keywords)}
    img_features = data[:NUM_IMAGES] / 255.0

    semantic_feat = np.zeros((NUM_IMAGES, len(all_keywords)))
    confidence_feat = np.zeros((NUM_IMAGES, len(all_keywords)))

    for i, kws in enumerate(semantic_features):
        for kw, score in zip(kws, [r['score'] for r in top_results]):
            semantic_feat[i][keyword_idx[kw]] = 1.0
            confidence_feat[i][keyword_idx[kw]] = score

    # 特征归一化
    img_features = (img_features - np.mean(img_features, axis=0)) / np.std(img_features, axis=0)
    node_feat = np.hstack([img_features, semantic_feat, confidence_feat]).astype(np.float32)

    # 图结构优化
    similarity_matrix = np.zeros((NUM_IMAGES, NUM_IMAGES))
    for i in range(NUM_IMAGES):
        for j in range(NUM_IMAGES):
            if i != j:
                similarity_matrix[i][j] = jaccard_similarity(
                    set(semantic_features[i]),
                    set(semantic_features[j])
                )

    # 动态阈值调整
    similarity_threshold = np.percentile(similarity_matrix, SIMILARITY_PERCENTILE)
    edges, edge_weights = [], []

    for i in range(NUM_IMAGES):
        for j in range(i + 1, NUM_IMAGES):
            if similarity_matrix[i][j] > max(similarity_threshold, 0.25):  # 混合阈值
                edges.extend([(i, j), (j, i)])
                edge_weights.extend([similarity_matrix[i][j]] * 2)

    # 保存图数据
    np.savez('graph_data_optimized.npz',
             node_feat=node_feat,
             edges=np.array(edges, dtype=np.int64),
             edge_weights=np.array(edge_weights, dtype=np.float32))


if __name__ == '__main__':
    main()