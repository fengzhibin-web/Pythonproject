import requests
import time
import csv
import numpy as np
import base64
import pickle
import json

# 百度API的认证信息
API_KEY = "6dLVZmgR8DOq8wAFXMSlf4bt"
SECRET_KEY = "ZEqigFIT7ufkIT5q9kjaoM3MkFadA219"


def unpickle(file):
    # 加载CIFAR-10数据集#
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def get_task_result(task_id):
    # 根据task_id获取问题结果#
    url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/image-understanding/get-result?access_token=" + get_access_token()

    payload = json.dumps({
        "task_id": task_id
    }, ensure_ascii=False)
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.post(url, headers=headers, data=payload.encode("utf-8"))
    if response.status_code == 200:
        return response.json()
    return None


def main():
    # 记录程序开始时间
    program_start_time = time.time()

    # 创建CSV文件并写入英文表头
    with open('cifar10_recognition_data_batch_1.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ['Image_Index', 'True_Label', 'Label_Name', 'Identify_Description', 'Identify_Result', 'IsCorrect',
             'API Processing Time'])

    # CIFAR-10数据集类别
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # 加载CIFAR-10数据集
    dataset_path = "cifar-10-batches-py/data_batch_1"
    try:
        batch = unpickle(dataset_path)
        data = batch[b'data']  # 图像数据
        labels = batch[b'labels']  # 图像标签
        print(f"成功加载数据集，图片数量: {len(data)}")
    except Exception as e:
        print(f"加载数据集失败: {e}")
        return

    # 处理每张图片
    for i in range(len(data)):
        # 获取图片数据并转换为RGB格式
        image_data = data[i].reshape(3, 32, 32).transpose(1, 2, 0)
        image_data = np.ascontiguousarray(image_data)

        # 确保图像数据类型为uint8
        if image_data.dtype != np.uint8:
            image_data = image_data.astype(np.uint8)

        # 将图片转换为jpg格式
        from PIL import Image
        img = Image.fromarray(image_data, 'RGB')
        img = img.resize((64, 64))  # 调整图片大小

        # 获取当前图片的标签
        true_label = labels[i]
        print(f"第{i + 1}张图片的真实标签true_label: {true_label}")

        # 将图片转换为base64
        import io
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)  # 控制图片质量
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # 检查base64编码后的大小
        if len(image_base64) > 10 * 1024 * 1024:  # 10MB
            print(f"第{i + 1}张图片base64编码后大小超过10MB")
            continue

        # 调用问题识别API
        url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/image-understanding/request?access_token=" + get_access_token()

        # 检查URL长度
        if len(url) > 1024:
            print(f"第{i + 1}张图片API URL长度超过1024字节")
            continue

        payload = json.dumps({
            "image": image_base64,  # 图片数据
            "question": "识别结果回答airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck其一，不需要分析，不回答其他多余内容"
        }, ensure_ascii=False)
        headers = {
            'Content-Type': 'application/json'
        }

        # 记录请求开始时间
        start_time = time.time()
        try:
            response = requests.post(url, headers=headers, data=payload.encode("utf-8"))
            response.raise_for_status()
            print(f"第{i + 1}张图片API调用成功")
        except Exception as e:
            print(f"第{i + 1}张图片API调用失败: {e}")
            continue

        # 记录请求结束时间并计算耗时
        end_time = time.time()
        request_duration = end_time - start_time

        # 处理响应
        if response.status_code == 200:
            try:
                result = response.json()
                print(f"第{i + 1}张图片API请求返回结果: {result}")

                if 'result' in result and 'task_id' in result['result']:
                    task_id = result['result']['task_id']
                    print(f"获取到的task_id: {task_id}")

                    # 获取问题结果
                    task_start_time = time.time()  # 记录任务开始时间
                    task_result = None
                    total_wait_time = 0  # 记录总等待时间

                    while True:
                        time.sleep(3)  # 每次等待3秒
                        total_wait_time += 3  # 累加等待时间
                        task_result = get_task_result(task_id)
                        task_end_time = time.time()  # 记录任务结束时间

                        # 判断重试条件
                        if task_result and 'result' in task_result and task_result['result'].get(
                                'ret_msg') == 'processing' and task_result['result'].get('ret_code') == 1:
                            print(f"第{i + 1}张图片处理中，继续等待...")
                        elif task_result and 'result' in task_result and task_result['result'].get('ret_code') == 0:
                            break  # 如果获取到最终结果，退出循环
                        else:
                            break  # 其他情况直接退出循环

                    if task_result:
                        print(f"第{i + 1}张图片问题处理结果: {task_result}")

                        # 计算实际耗时（减去等待时间和重试消耗的时间）
                        task_duration = (task_end_time - task_start_time - total_wait_time) + request_duration
                        print(f"第{i + 1}张图片API处理请求的总耗时: {task_duration:.2f}秒")
                        print(f"初始请求耗时: {request_duration:.2f}秒")
                        print(f"任务处理耗时: {(task_end_time - task_start_time - total_wait_time):.2f}秒")

                        # 获取并打印description字段
                        if 'result' in task_result and 'description' in task_result['result']:
                            description = task_result['result']['description']
                            print(f"第{i + 1}张图片的识别结果description: {description}")

                            # 判断识别结果对应的label_names索引
                            label_names = [
                                "airplane",  # label_names[0]
                                "automobile",  # label_names[1]
                                "bird",  # label_names[2]
                                "cat",  # label_names[3]
                                "deer",  # label_names[4]
                                "dog",  # label_names[5]
                                "frog",  # label_names[6]
                                "horse",  # label_names[7]
                                "ship",  # label_names[8]
                                "truck"  # label_names[9]
                            ]

                            # 获取真实标签对应的名称
                            label_name = label_names[true_label] if true_label < len(label_names) else "Unknown"

                            # 获取识别结果对应的索引
                            identify_result = label_names.index(description) if description in label_names else -1
                            print(f"第{i + 1}张图片的识别结果编号identify result: {identify_result}")

                            # 判断识别结果是否正确
                            is_correct = identify_result == true_label
                            if is_correct:
                                print("识别结果为识别正确")
                            else:
                                print("识别结果为识别错误")

                            # 打印空行
                            print()

                            # 将结果写入CSV文件
                            with open('cifar10_recognition_data_batch_1.csv', 'a', newline='',
                                      encoding='utf-8') as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerow([
                                    f"Image {i + 1}",
                                    true_label,
                                    label_name,
                                    description,
                                    identify_result,
                                    is_correct,
                                    f"{task_duration:.2f} seconds"
                                ])

                    else:
                        print("无法获取问题结果")
                else:
                    print("返回结果中未包含task_id")
            except json.JSONDecodeError:
                print("返回结果不是有效的JSON格式")
        else:
            print(f"请求失败，状态码: {response.status_code}, 响应内容: {response.text}")

        time.sleep(1)  # 每次请求后等待1秒


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
