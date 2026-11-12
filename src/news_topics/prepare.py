"""Download versioned public dataset and encoder assets, recording their checksums."""
import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import requests

DATA_REVISION = "03836ce08fce38daac53e9b4255ef16e0b9f6303"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DATA_BASE = f"https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/{DATA_REVISION}/data/ag_news_csv"
MODEL_BASE = f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}"


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with requests.get(url, stream=True, timeout=(15, 120)) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                handle.write(chunk)
    temporary.replace(path)
    return {"url": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size}


def prepare(root):
    root = Path(root)
    assets = {}
    for name in ("train.csv", "test.csv", "readme.txt", "classes.txt"):
        print(f"Downloading AG News {name}", flush=True)
        assets[f"data/{name}"] = download(f"{DATA_BASE}/{name}", root / "data" / name)
    for remote, local in [("onnx/model.onnx", "encoder.onnx"),
                          ("tokenizer.json", "tokenizer.json"),
                          ("README.md", "MODEL_CARD.md"), ("config.json", "config.json")]:
        print(f"Downloading MiniLM {remote}", flush=True)
        assets[f"models/minilm/{local}"] = download(f"{MODEL_BASE}/{remote}",
                                                   root / "models/minilm" / local)
    manifest = {"dataset": "AG News topic classification, version 3",
                "dataset_revision": DATA_REVISION, "model": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "retrieved_at_utc": datetime.now(UTC).isoformat(), "assets": assets}
    (root / "provenance.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    prepare(parser.parse_args().root)


if __name__ == "__main__":
    main()

