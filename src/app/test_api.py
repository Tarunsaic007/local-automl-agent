"""Integration tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_submit_valid_csv(sample_csv_bytes):
    response = client.post(
        "/submit",
        files={"file": ("test.csv", sample_csv_bytes, "text/csv")},
    )
    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "queued"


def test_submit_non_csv_rejected():
    response = client.post(
        "/submit",
        files={"file": ("test.txt", b"not a csv", "text/plain")},
    )
    assert response.status_code == 400


def test_submit_empty_file_rejected():
    response = client.post(
        "/submit",
        files={"file": ("empty.csv", b"", "text/csv")},
    )
    assert response.status_code == 400


def test_status_not_found():
    response = client.get("/status/nonexistent-job")
    assert response.status_code == 404


def test_status_found_after_submit(sample_csv_bytes):
    submit = client.post(
        "/submit",
        files={"file": ("test.csv", sample_csv_bytes, "text/csv")},
    )
    job_id = submit.json()["job_id"]

    status = client.get(f"/status/{job_id}")
    assert status.status_code == 200
    data = status.json()
    assert data["job_id"] == job_id
    assert "status" in data
    assert "progress" in data


def test_list_jobs(sample_csv_bytes):
    client.post("/submit", files={"file": ("test.csv", sample_csv_bytes, "text/csv")})
    response = client.get("/jobs")
    assert response.status_code == 200
    assert "jobs" in response.json()
    assert len(response.json()["jobs"]) >= 1