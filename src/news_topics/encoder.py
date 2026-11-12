"""Frozen MiniLM encoder: masked mean pooling and L2 normalisation on CPU."""
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


class Encoder:
    def __init__(self, directory, threads=2):
        directory = Path(directory)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(directory / "encoder.onnx"), options,
                                            providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

    def encode(self, texts, batch_size=32):
        result = []
        for start in range(0, len(texts), batch_size):
            batch = self.tokenizer.encode_batch(texts[start:start + batch_size])
            inputs = {"input_ids": np.array([x.ids for x in batch], dtype=np.int64),
                      "attention_mask": np.array([x.attention_mask for x in batch], dtype=np.int64),
                      "token_type_ids": np.array([x.type_ids for x in batch], dtype=np.int64)}
            needed = {x.name for x in self.session.get_inputs()}
            tokens = self.session.run(None, {k: v for k, v in inputs.items() if k in needed})[0]
            mask = inputs["attention_mask"][..., None]
            pooled = (tokens * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1)
            pooled /= np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-12)
            result.append(pooled.astype(np.float32))
            if start % (batch_size * 25) == 0:
                print(f"Encoded {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
        return np.concatenate(result)
