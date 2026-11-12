"""Behaviour checks use a small excerpt of the published AG News training file."""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from news_topics.api import Predictor, create_app
from news_topics.data import read_news, split_data
from news_topics.encoder import Encoder
from news_topics.train import make_baseline

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/ag_news.csv"


def test_official_labels_and_row_ids():
    frame = read_news(FIXTURE)
    assert len(frame) == 64
    assert set(frame.label) == {0, 1, 2, 3}
    assert frame.row_id.tolist() == list(range(1, 65))
    assert frame.text_hash.str.len().eq(64).all()


def test_no_exact_text_leakage_between_splits():
    frame = read_news(FIXTURE)
    held_out = frame.groupby("label").tail(2)
    # Deliberately pass the original rows in both files to check overlap removal.
    fit, validation, audit = split_data(pd.concat([frame, frame.head(1)]), held_out, limit=0)
    assert not set(fit.text_hash) & set(validation.text_hash)
    assert not set(fit.text_hash) & set(held_out.text_hash)
    assert not set(validation.text_hash) & set(held_out.text_hash)
    assert audit["duplicate_training_texts_removed"] == 1
    assert audit["training_texts_matching_test_removed"] == len(held_out)
    assert len(fit) + len(validation) == 56


def test_baseline_predicts_from_actual_held_out_articles():
    frame = read_news(FIXTURE)
    test = frame.groupby("label").tail(2)
    train = frame[~frame.text_hash.isin(test.text_hash)]
    model = make_baseline()
    model.fit(train.text, train.label)
    probabilities = model.predict_proba(test.text)
    assert probabilities.shape == (8, 4)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    assert not set(test.text_hash) & set(train.text_hash)
    # This tiny fixture is not a performance benchmark.


def test_api_handles_missing_models_and_validates_input(tmp_path):
    (tmp_path / "data").mkdir()
    shutil.copyfile(FIXTURE, tmp_path / "data/test.csv")
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/api/health").status_code == 503
        assert client.post("/api/predict", json={"text": "   "}).status_code == 422
        article = read_news(FIXTURE).text.iloc[0]
        assert client.post("/api/predict", json={"text": article}).status_code == 503
        examples = client.get("/api/examples").json()["examples"]
        assert len(examples) == 4
        assert all(x["text"] in set(read_news(FIXTURE).text) for x in examples)


def test_report_metrics_match_exported_test_predictions():
    path = ROOT / "reports/test_predictions.csv"
    if not path.exists():
        pytest.skip("Generate reports before checking the saved full evaluation")
    rows = pd.read_csv(path)
    report = json.loads((ROOT / "reports/metrics.json").read_text())
    assignments = pd.read_csv(ROOT / "reports/training_assignments.csv")
    assert len(rows) == 7600
    assert assignments.row_id.is_unique
    fit = assignments[assignments.split == "train"]
    validation = assignments[assignments.split == "validation"]
    assert not set(fit.row_id) & set(validation.row_id)
    if (ROOT / "data/train.csv").exists():
        source = read_news(ROOT / "data/train.csv")
        selected = source[source.row_id.isin(assignments.row_id)]
        assert len(selected) == len(assignments)
        assert not set(rows.text_hash) & set(selected.text_hash)
    for method in ("tfidf", "minilm"):
        assert rows[f"{method}_prediction"].eq(rows.label).mean() == pytest.approx(
            report["test"][method]["accuracy"])
        assert np.array(report["test"][method]["confusion_matrix"]).sum() == 7600


@pytest.mark.skipif(not (ROOT / "models/minilm/encoder.onnx").exists(),
                    reason="Run preparation for the local encoder integration test")
def test_encoder_batch_padding_does_not_change_article_embedding():
    texts = read_news(FIXTURE).text.tolist()
    encoder = Encoder(ROOT / "models/minilm")
    alone = encoder.encode([texts[0]])
    batch = encoder.encode([texts[0], max(texts, key=len)])
    assert alone.shape == (1, 384)
    assert np.isfinite(batch).all()
    np.testing.assert_allclose(np.linalg.norm(batch, axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(alone[0], batch[0], atol=1e-5)


@pytest.mark.skipif(not (ROOT / "models/minilm_classifier.joblib").exists()
                    or not (ROOT / "reports/test_predictions.csv").exists(),
                    reason="Run training for the local prediction integration test")
def test_live_predictions_match_saved_evaluation():
    test = read_news(ROOT / "data/test.csv")
    saved = pd.read_csv(ROOT / "reports/test_predictions.csv")
    predictor = Predictor(ROOT)
    for label in range(4):
        row = test[test.label == label].iloc[0]
        result = predictor.predict(row.text)
        expected = saved[saved.row_id == row.row_id].iloc[0]
        from news_topics.data import LABELS
        for method in ("tfidf", "minilm"):
            assert result["models"][method]["topic"] == LABELS[int(expected[f"{method}_prediction"])]
            assert sum(x["score"] for x in result["models"][method]["scores"]) == pytest.approx(1)

