import requests
import os
import base64
import time
import pickle
import numpy as np

API_KEY = "81wYDARK0gL1o8x6meQria0C"
SECRET_KEY = "g1HLzXMPMderxLILDM1Tyj5GoQY9oKqR"


def unpickle(file):
    """加载CIFAR-10数据集的一个batch"""
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def main():
    url = "https://aip.baidubce.com/rest/2.0/image-classify/v2/advanced_general?access_token=" + get_access_token()

    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json'
    }

    # 在这里填写CIFAR-10数据集的本地路径
    dataset_path = "C:/Users/13525/Desktop/cifar-10-batches-py/data_batch_1"  # 修改为你的实际路径

    # 使用unpickle函数加载CIFAR-10数据集
    batch = unpickle(dataset_path)
    labels = batch[b'labels']
    data = batch[b'data']

    # 统计变量
    total = 0
    correct = 0

    # 处理数据集中的图片
    for i in range(len(data)):
        # 获取图片数据并转换为RGB格式
        image_data = data[i].reshape(3, 32, 32).transpose(1, 2, 0)

        # 确保数组是C连续的
        image_data = np.ascontiguousarray(image_data)

        # 将图片转换为base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        # 构建请求
        payload = f'image={image_base64}'
        response = requests.post(url, headers=headers, data=payload)

        # 处理响应
        if response.status_code == 200:
            result = response.json()
            if result.get('result'):
                top_result = result['result'][0]['keyword']
                # 获取真实标签（需要根据CIFAR-10标签映射）
                true_label = get_cifar10_label(labels[i])
                if top_result.lower() == true_label.lower():
                    correct += 1
            total += 1

        time.sleep(1)  # 添加延时避免QPS超限

    # 输出最终准确率
    print(f"测试完成，准确率: {correct / total:.2%}")


def get_cifar10_label(index):
    """将CIFAR-10标签索引转换为类别名称"""
    cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck']
    return cifar10_labels[index]


def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))


if __name__ == '__main__':
    main()
