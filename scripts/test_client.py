from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

response = client.post(
    "/api/v1/query",
    json={"query": "give me assigment overview", "top_k": 5}
)
print("Status Code:", response.status_code)
print("Response JSON:", response.json())
