
# encoding:utf-8

import requests

'''
图像识别组合API
'''

request_url = "https://aip.baidubce.com/api/v1/solution/direct/imagerecognition/combination"

import base64

# 先定义headers和access_token
access_token = '24.ad065e880cedef7edbf1aaff18bbdd1d.2592000.1743778252.282335-117847383'
headers = {'content-type': 'application/json'}
request_url = request_url + "?access_token=" + access_token

# 读取图片并转换为base64
with open(r'C:\Users\13525\Desktop\test2.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

# 构建请求参数
params = {
    "image": image_data,
    "scenes": ["animal","plant","ingredient","dishs", "red_wine","currency","landmark"],
    "sceneConf": {
    }
}

# 将params转换为JSON字符串
import json
params_json = json.dumps(params)

# 发送请求
response = requests.post(request_url, data=params_json, headers=headers)

# 处理响应
# ... 其他代码 ...

# 处理响应
if response:
    result = response.json()
    print("识别结果：")
    for key, value in result.items():
        print(f"{key}:")
        if isinstance(value, list):
            for item in value:
                print(f"  {item}")
        else:
            print(f"  {value}")