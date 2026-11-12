"""Read AG News, retaining original file row numbers and exact-text grouping."""
import hashlib
import html
import re

import pandas as pd
from sklearn.model_selection import train_test_split

LABELS = ["World", "Sports", "Business", "Science and Technology"]


def clean_text(text):
    text = html.unescape(text).replace("\\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def read_news(path):
    frame = pd.read_csv(path, header=None, names=["label", "title", "description"],
                        keep_default_na=False)
    if frame.empty or not frame.label.isin([1, 2, 3, 4]).all():
        raise ValueError("Expected AG News labels 1 through 4")
    frame["row_id"] = range(1, len(frame) + 1)
    frame["text"] = (frame.title + " " + frame.description).map(clean_text)
    if frame.text.eq("").any():
        raise ValueError("Empty articles in dataset")
    frame["text_hash"] = frame.text.str.casefold().map(
        lambda text: hashlib.sha256(text.encode()).hexdigest())
    frame["label"] = frame.label - 1
    return frame


def split_data(train, test, limit=20000, seed=42):
    """Keep official test intact; remove matching training texts before sampling."""
    unique = train.drop_duplicates("text_hash", keep="first")
    clean = unique[~unique.text_hash.isin(set(test.text_hash))].copy()
    if limit and limit < len(clean):
        clean, _ = train_test_split(clean, train_size=limit, stratify=clean.label,
                                    random_state=seed)
    fit, validation = train_test_split(clean, test_size=.2, stratify=clean.label,
                                       random_state=seed)
    audit = {"official_training_rows": len(train), "official_test_rows": len(test),
             "duplicate_training_texts_removed": len(train) - len(unique),
             "training_texts_matching_test_removed": len(unique) - len(
                 unique[~unique.text_hash.isin(set(test.text_hash))]),
             "training_rows": len(fit), "validation_rows": len(validation),
             "test_duplicate_texts": int(test.text_hash.duplicated().sum()),
             "seed": seed, "requested_training_pool_limit": limit,
             "split": "Stratified train/validation from official training; official test unchanged"}
    return fit.reset_index(drop=True), validation.reset_index(drop=True), audit

