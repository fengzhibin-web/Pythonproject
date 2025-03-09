import requests
import json

API_KEY = "81wYDARK0gL1o8x6meQria0C"
SECRET_KEY = "g1HLzXMPMderxLILDM1Tyj5GoQY9oKqR"


def main():
    url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/image-understanding/request?access_token=" + get_access_token()

    payload = json.dumps({
        "url": "https://qcloud.dpfile.com/pc/oWkmtPhfzOiRKwKJnjy8yE4Cjk8t3FmukMlwDirj8x5lz5pEXaAjfdXiOZWKlHBN.jpg",
        "question": "图上有什么？"
    }, ensure_ascii=False)
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload.encode("utf-8"))

    print(response.text)


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
