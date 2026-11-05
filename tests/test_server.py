"""The standalone web app (server.py) must stay in step with app.run_batch."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import server

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "judge_test"
SAMPLE = ROOT / "data" / "sample" / "test_sample.csv"
client = TestClient(server.api)


def test_batch_returns_checks_and_efficiency_without_answers():
    with SAMPLE.open("rb") as f:
        r = client.post("/api/batch", files={"test": ("test.csv", f, "text/csv")})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["predictions"].startswith("station_id,date,discharge_cfs")
    assert "Submission checks: passed" in d["metrics"] and "8 · Efficiency" in d["metrics"]


@pytest.mark.skipif(not (KIT / "answers.csv").exists(), reason="practice kit not built")
def test_batch_with_answers_returns_all_metric_families():
    with (KIT / "test.csv").open("rb") as t, (KIT / "answers.csv").open("rb") as a:
        r = client.post("/api/batch", files={"test": ("test.csv", t, "text/csv"), "answers": ("answers.csv", a, "text/csv")},
                        data={"robustness": "true"})
    assert r.status_code == 200, r.text
    assert "rows matched" in r.json()["score"]
    assert "7 · Robustness" in r.json()["metrics"]


def test_batch_rejects_repeated_keys_with_clear_message(tmp_path):
    bad = tmp_path / "dup.csv"
    bad.write_text("station_id,date\n01646500,2025-11-01\n01646500,2025-11-01\n", encoding="utf-8")
    with bad.open("rb") as f:
        r = client.post("/api/batch", files={"test": ("test.csv", f, "text/csv")})
    assert r.status_code == 400 and "repeats 1 station_id/date pair" in r.json()["detail"]


def test_page_has_remove_buttons_for_each_file():
    html = client.get("/").text
    for field in ("f_test", "f_train", "f_answers"):
        assert f'data-for="{field}"' in html
    assert 'id="clearFiles"' in html


@pytest.mark.skipif(not (KIT / "answers.csv").exists(), reason="practice kit not built")
def test_answers_uploaded_as_train_is_rejected():
    with (KIT / "test.csv").open("rb") as t, (KIT / "answers.csv").open("rb") as a:
        r = client.post("/api/batch", files={"test": ("test.csv", t, "text/csv"), "train": ("answers.csv", a, "text/csv")})
    assert r.status_code == 400 and "upload it in the answers box" in r.json()["detail"]
