"""Minimal offline verification for PLC Anomaly Lab."""
from app import app

client = app.test_client()
assert client.get("/health").status_code == 200
assert client.get("/api/metadata").status_code == 200

response = client.post("/api/run", json={"seed": 2026, "sets": [1, 2, 3, 4, 5]})
assert response.status_code == 200, response.get_json()
payload = response.get_json()
assert len(payload["results"]) == 5
assert all(len(entry["models"]) == 3 for entry in payload["results"])

assert client.post("/api/inspect", json={"seed": 2026, "scenario": 6}).status_code == 200
assert client.get("/api/download/test-set-5.csv?seed=2026").status_code == 200
print("OK — health, metadata, training, inspection and CSV export verified.")
