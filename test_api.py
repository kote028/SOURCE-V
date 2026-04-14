import requests

url = "http://127.0.0.1:5000/analyze"

files = {
    "video": open("data/train/fake/07_03__hugging_happy__7NGMD8FT.mp4", "rb")
}

res = requests.post(url, files=files)

print(res.json())