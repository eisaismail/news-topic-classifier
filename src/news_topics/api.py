"""Local browser interface and prediction API. Submitted text is not retained."""
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .data import LABELS, clean_text, read_news
from .encoder import Encoder

STATIC = Path(__file__).parent / "static"


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)

    @field_validator("text")
    @classmethod
    def nonempty(cls, value):
        value = clean_text(value)
        if not value:
            raise ValueError("Enter a headline or short article")
        return value


class Predictor:
    def __init__(self, root):
        # Only load model files generated locally by news_topics.train.
        self.baseline = joblib.load(root / "models/tfidf.joblib")
        self.classifier = joblib.load(root / "models/minilm_classifier.joblib")
        self.encoder = Encoder(root / "models/minilm")

    def predict(self, text):
        probabilities = {"tfidf": self.baseline.predict_proba([text])[0],
                         "minilm": self.classifier.predict_proba(self.encoder.encode([text]))[0]}
        results = {}
        for name, values in probabilities.items():
            results[name] = {"topic": LABELS[int(values.argmax())],
                             "scores": [{"topic": label, "score": float(value)}
                                        for label, value in zip(LABELS, values)]}
        return {"models": results, "agree": results["tfidf"]["topic"] == results["minilm"]["topic"],
                "notice": "Scores are not calibrated certainty. Only four English news topics are supported."}


def create_app(root=None):
    root = Path(root or os.getenv("NEWS_PROJECT_ROOT", ".")).resolve()

    @asynccontextmanager
    async def lifespan(app):
        app.state.predictor = None
        try:
            app.state.predictor = Predictor(root)
        except FileNotFoundError:
            pass
        yield

    app = FastAPI(title="News Topic Classifier", version="1.0.0", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    def health():
        if app.state.predictor is None:
            raise HTTPException(503, "Run preparation and training from the project folder first.")
        return {"status": "ready", "models": ["tfidf", "minilm"]}

    @app.post("/api/predict")
    def predict(body: PredictionRequest):
        if app.state.predictor is None:
            raise HTTPException(503, "Models are missing. Run preparation and training first.")
        return app.state.predictor.predict(body.text)

    @app.get("/api/evaluation")
    def evaluation():
        path = root / "reports/metrics.json"
        if not path.exists():
            raise HTTPException(503, "Evaluation has not been generated. Run training first.")
        return json.loads(path.read_text(encoding="utf-8"))

    @app.get("/api/examples")
    def examples():
        if not (root / "data/test.csv").exists():
            raise HTTPException(503, "AG News test file is missing. Run preparation first.")
        test = read_news(root / "data/test.csv")
        chosen = test.groupby("label", sort=True).head(1)
        result = [{"row_id": int(row.row_id), "topic": LABELS[int(row.label)],
                   "text": row.text, "kind": "Dataset example"} for row in chosen.itertuples()]
        errors_path = root / "reports/errors.csv"
        if errors_path.exists():
            errors = pd.read_csv(errors_path)
            for row in errors[errors.method == "minilm"].groupby("actual").head(1).itertuples():
                item = test.loc[test.row_id == row.test_row_id].iloc[0]
                result.append({"row_id": int(item.row_id), "topic": row.actual,
                               "text": item.text, "kind": "Model mistake",
                               "predicted": row.predicted})
        return {"source": "AG News official test.csv; row IDs are 1-based", "examples": result}

    return app


app = create_app()

