import urllib.request
import json
url = "http://127.0.0.1:8000/api/v1/query"
data = json.dumps({"query": "give me assigment overview", "top_k": 5}).encode('utf-8')
req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'}, method='POST')
try:
    with urllib.request.urlopen(req) as response:
        print(response.read().decode('utf-8'))
except Exception as e:
    print(e)
