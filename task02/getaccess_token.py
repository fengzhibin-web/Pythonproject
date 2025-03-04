import requests
import json


def main():
    url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=81wYDARK0gL1o8x6meQria0C&client_secret=g1HLzXMPMderxLILDM1Tyj5GoQY9oKqR"

    payload = ""
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)


if __name__ == '__main__':
    main()
