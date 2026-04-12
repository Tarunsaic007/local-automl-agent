import pytest 
from fastapi.testclient import TestClient
from app.main import app
from app.store import JobStore


@pytest.fixture(autouse=True)
def reset_job_store():
    """Reset the job store singleton between tests to prevent state leakage."""
    JobStore._instance = None
    yield
    JobStore._instance = None


@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def sample_csv_bytes():
    """A minimal classification CSV for testing."""
    return b"""age,salary,department,churn
25,50000,Engineering,0
35,80000,Marketing,1
45,120000,Engineering,0
28,60000,Sales,1
52,150000,Marketing,0
"""