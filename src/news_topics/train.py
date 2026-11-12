"""Train comparable topic classifiers and evaluate on the official AG News test set."""
import argparse
import hashlib
import json
import time
from importlib.metadata import version
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from .data import LABELS, read_news, split_data
from .encoder import Encoder


def make_baseline(c=4):
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000,
                                  sublinear_tf=True, strip_accents="unicode")),
        ("classifier", LogisticRegression(C=c, max_iter=400, random_state=42)),
    ])


def cached_embeddings(frame, encoder, root, name, identity):
    digest = hashlib.sha256((identity + "|".join(frame.text_hash)).encode()).hexdigest()
    path = root / "cache" / f"{name}_{digest[:16]}.npy"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        return np.load(path, allow_pickle=False)
    matrix = encoder.encode(frame.text.tolist())
    np.save(path, matrix, allow_pickle=False)
    return matrix


def metrics(labels, probabilities):
    predicted = probabilities.argmax(axis=1)
    return {"accuracy": float(accuracy_score(labels, predicted)),
            "macro_f1": float(f1_score(labels, predicted, average="macro")),
            "per_class": classification_report(labels, predicted, labels=list(range(4)),
                                                target_names=LABELS, output_dict=True,
                                                zero_division=0),
            "confusion_matrix": confusion_matrix(labels, predicted, labels=list(range(4))).tolist()}


def paired_accuracy_interval(labels, baseline, minilm):
    differences = (minilm == labels).astype(float) - (baseline == labels).astype(float)
    rng = np.random.default_rng(42)
    means = [float(differences[rng.integers(0, len(labels), len(labels))].mean())
             for _ in range(1000)]
    return {"difference_minilm_minus_tfidf": float(differences.mean()),
            "bootstrap_95_percent_interval": np.quantile(means, [.025, .975]).tolist(),
            "resamples": 1000,
            "caveat": "Paired row bootstrap; related news stories are not independent."}


def train(root, limit=20000):
    root = Path(root)
    start = time.perf_counter()
    provenance = json.loads((root / "provenance.json").read_text())
    for name in ("data/train.csv", "data/test.csv", "models/minilm/encoder.onnx",
                 "models/minilm/tokenizer.json"):
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if actual != provenance["assets"][name]["sha256"]:
            raise ValueError(f"Checksum mismatch: {name}. Run preparation again.")
    official_train = read_news(root / "data/train.csv")
    test = read_news(root / "data/test.csv")
    fit, validation, audit = split_data(official_train, test, limit)
    print(json.dumps(audit), flush=True)
    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    candidates = []
    best_baseline, best_score = None, -1
    for c in (1, 4, 16):
        model = make_baseline(c)
        model.fit(fit.text, fit.label)
        score = f1_score(validation.label, model.predict(validation.text), average="macro")
        candidates.append({"method": "tfidf", "C": c, "validation_macro_f1": float(score)})
        print(f"TF-IDF C={c}: validation macro F1 {score:.4f}", flush=True)
        if score > best_score:
            best_baseline, best_score = model, score
    joblib.dump(best_baseline, root / "models/tfidf.joblib")
    encoder = Encoder(root / "models/minilm")
    identity = provenance["model_revision"] + provenance["assets"]["models/minilm/encoder.onnx"]["sha256"]
    embeddings = {}
    for name, frame in (("train", fit), ("validation", validation)):
        embeddings[name] = cached_embeddings(frame, encoder, root, name, identity)
    best_minilm, best_score = None, -1
    for c in (1, 4, 16):
        model = LogisticRegression(C=c, max_iter=400, random_state=42)
        model.fit(embeddings["train"], fit.label)
        score = f1_score(validation.label, model.predict(embeddings["validation"]), average="macro")
        candidates.append({"method": "minilm", "C": c, "validation_macro_f1": float(score)})
        print(f"MiniLM C={c}: validation macro F1 {score:.4f}", flush=True)
        if score > best_score:
            best_minilm, best_score = model, score
    joblib.dump(best_minilm, root / "models/minilm_classifier.joblib")
    # Neither classifier is refitted or selected using the test labels.
    test_embeddings = cached_embeddings(test, encoder, root, "test", identity)
    probabilities = {"tfidf": best_baseline.predict_proba(test.text),
                     "minilm": best_minilm.predict_proba(test_embeddings)}
    results = {name: metrics(test.label, values) for name, values in probabilities.items()}
    labels = test.label.to_numpy()
    report = {"dataset": "AG News", "labels": LABELS, "split": audit,
              "selection": "Highest validation macro F1 from C=1,4,16; no test tuning or refitting",
              "selected_C": {"tfidf": best_baseline.named_steps["classifier"].C,
                             "minilm": best_minilm.C},
              "validation_candidates": candidates, "test": results,
              "paired_accuracy_comparison": paired_accuracy_interval(
                  labels, probabilities["tfidf"].argmax(1), probabilities["minilm"].argmax(1)),
              "encoder": {"model": provenance["model"], "revision": provenance["model_revision"],
                          "frozen": True, "max_tokens": 256, "pooling": "Masked mean then L2",
                          "runtime": "ONNX CPU, float32 weights, 2 intra-op threads"},
              "package_versions": {name: version(name) for name in
                                    ("numpy", "pandas", "scikit-learn", "onnxruntime", "tokenizers")},
              "elapsed_seconds": round(time.perf_counter() - start, 2)}
    (reports / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    rows = test[["row_id", "text_hash", "label"]].copy()
    for name, values in probabilities.items():
        rows[f"{name}_prediction"] = values.argmax(axis=1)
        rows[f"{name}_score"] = values.max(axis=1)
    rows.to_csv(reports / "test_predictions.csv", index=False)
    assignments = pd.concat([fit.assign(split="train"), validation.assign(split="validation")])
    assignments[["row_id", "label", "split"]].to_csv(
        reports / "training_assignments.csv", index=False)
    error_rows = []
    for name in probabilities:
        errors = rows[rows[f"{name}_prediction"] != rows.label]
        for label in range(4):
            for _, row in errors[errors.label == label].sort_values(
                    f"{name}_score", ascending=False).head(3).iterrows():
                error_rows.append({"method": name, "test_row_id": int(row.row_id),
                                   "actual": LABELS[label],
                                   "predicted": LABELS[int(row[f"{name}_prediction"])],
                                   "model_score": float(row[f"{name}_score"])})
    pd.DataFrame(error_rows).to_csv(reports / "errors.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, result) in zip(axes, results.items()):
        matrix = np.array(result["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha="center", va="center",
                    color="white" if value > matrix.max() / 2 else "#172b42")
        short = ["World", "Sports", "Business", "Sci/Tech"]
        ax.set(xticks=range(4), yticks=range(4), xticklabels=short, yticklabels=short,
               xlabel="Predicted topic", ylabel="Dataset topic",
               title=f"{'TF-IDF' if name == 'tfidf' else 'MiniLM'}: {result['accuracy']:.1%} accuracy")
    fig.suptitle("AG News official test set: 7,600 articles")
    fig.tight_layout()
    fig.savefig(reports / "confusion_matrix.svg")
    plt.close(fig)
    print(json.dumps({name: {k: v for k, v in result.items() if k in ("accuracy", "macro_f1")}
                      for name, result in results.items()}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--train-limit", type=int, default=20000,
                        help="Pool size before 80/20 train/validation split; 0 uses all clean training")
    args = parser.parse_args()
    if args.train_limit < 0:
        parser.error("--train-limit must be nonnegative")
    train(args.root, args.train_limit)


if __name__ == "__main__":
    main()
