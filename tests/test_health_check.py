from fastapi.testclient import TestClient

from src.host.app import app

client = TestClient(app)


def test_liveness() -> None:
    response = client.get(
        "/api/v1/health_check/"
    )

    assert response.status_code == 200

    assert response.json() == {
        "status": "ok",
        "service": "speech-to-text",
        "version": "0.1.0",
        "environment": "development",
    }


def test_readiness_with_local_postgres() -> None:
    response = client.get(
        "/api/v1/health_check/ready"
    )

    assert response.status_code == 200

    assert response.json() == {
        "status": "ok",
        "database": "ok",
    }