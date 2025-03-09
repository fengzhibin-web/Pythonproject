
# encoding:utf-8

import requests
import base64

'''
通用物体和场景识别
'''

request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v2/advanced_general"
# 二进制方式打开图片文件
f = open('C:\\Users\\13525\\Desktop\\test.jpg', 'rb')
img = base64.b64encode(f.read())

params = {"image":img}
access_token = '24.570148c1d29ddc6336591a8bac62e473.2592000.1743695225.282335-117847383'
request_url = request_url + "?access_token=" + access_token
headers = {'content-type': 'application/x-www-form-urlencoded'}
response = requests.post(request_url, data=params, headers=headers)
if response:
    print (response.json())
