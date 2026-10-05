from __future__ import annotations

from fastapi.testclient import TestClient


def _create_prompt(client: TestClient, name: str = "support-email") -> int:
    response = client.post(
        "/prompts",
        json={"name": name, "actor": "alice", "reason": "create"},
    )
    assert response.status_code == 200, response.text
    return int(response.json()["id"])


def _add_version(client: TestClient, prompt_id: int, message: str) -> dict[str, object]:
    response = client.post(
        f"/prompts/{prompt_id}/versions",
        json={
            "system_prompt": f"Classify tickets. {message}",
            "model": "claude-haiku-4-5-20251001",
            "temperature": 0.0,
            "commit_message": message,
            "actor": "alice",
            "reason": f"add {message}",
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_new_versions_auto_increment(client: TestClient) -> None:
    prompt_id = _create_prompt(client)
    first = _add_version(client, prompt_id, "v1")
    second = _add_version(client, prompt_id, "v2")
    assert first["version"] == 1
    assert second["version"] == 2
    listed = client.get(f"/prompts/{prompt_id}/versions")
    assert listed.status_code == 200
    assert [row["version"] for row in listed.json()] == [1, 2]
    one = client.get(f"/prompts/{prompt_id}/versions/2")
    assert one.status_code == 200
    assert one.json()["commit_message"] == "v2"


def test_activating_older_version_rolls_back(client: TestClient) -> None:
    prompt_id = _create_prompt(client)
    v1 = _add_version(client, prompt_id, "v1")
    v2 = _add_version(client, prompt_id, "v2")
    live = client.post(
        f"/prompts/{prompt_id}/activate",
        json={"version": 2, "actor": "alice", "reason": "ship"},
    )
    assert live.status_code == 200
    assert live.json()["version_id"] == v2["id"]
    rolled = client.post(
        f"/prompts/{prompt_id}/activate",
        json={"version": 1, "actor": "alice", "reason": "rollback"},
    )
    assert rolled.status_code == 200
    assert rolled.json()["version"] == 1
    assert rolled.json()["version_id"] == v1["id"]


def test_activation_writes_audit_row(client: TestClient) -> None:
    prompt_id = _create_prompt(client)
    _add_version(client, prompt_id, "v1")
    client.post(
        f"/prompts/{prompt_id}/activate",
        json={"version": 1, "actor": "carol", "reason": "production"},
    )
    log = client.get("/audit-log")
    assert log.status_code == 200
    activates = [row for row in log.json() if row["action"] == "activate_version"]
    assert len(activates) == 1
    assert activates[0]["actor"] == "carol"
    assert activates[0]["details"]["reason"] == "production"
    assert activates[0]["prompt_id"] == prompt_id
