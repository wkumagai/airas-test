import os
from typing import Dict, Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets import load_dataset, Dataset as HFDataset


def _guess_text_fields(dataset_name: str):
    # ccdv arxiv/pubmed use 'article' and 'abstract'
    return "article", "abstract"


def _subsample(ds: HFDataset, n: Optional[int], seed: int) -> HFDataset:
    if n is None:
        return ds
    n = int(n)
    if n <= 0 or n >= len(ds):
        return ds
    return ds.shuffle(seed=seed).select(range(n))


def build_datasets(cfg, tokenizer) -> Dict[str, Dataset]:
    name = str(cfg.dataset.name)
    cache_dir = ".cache/"
    splits = cfg.dataset.splits

    raw = load_dataset(name, cache_dir=cache_dir)

    src_field, tgt_field = _guess_text_fields(name)

    max_encoder_len = int(cfg.dataset.preprocessing.max_encoder_len)
    max_decoder_len = int(cfg.dataset.preprocessing.max_decoder_len)
    truncate = bool(cfg.dataset.preprocessing.truncate)
    pad_to_max = bool(cfg.dataset.preprocessing.pad_to_max_length)

    def tok_fn(ex: Dict[str, Any]) -> Dict[str, Any]:
        src = ex.get(src_field, "")
        tgt = ex.get(tgt_field, "")

        enc = tokenizer(
            src,
            max_length=max_encoder_len,
            truncation=truncate,
            padding="max_length" if pad_to_max else False,
            return_attention_mask=True,
        )
        dec = tokenizer(
            tgt,
            max_length=max_decoder_len,
            truncation=True,
            padding="max_length" if pad_to_max else False,
            return_attention_mask=False,
        )

        labels = np.array(dec["input_ids"], dtype=np.int64)
        labels = np.where(labels == tokenizer.pad_token_id, -100, labels)

        return {
            "input_ids": np.array(enc["input_ids"], dtype=np.int64),
            "attention_mask": np.array(enc["attention_mask"], dtype=np.int64),
            "labels": labels,
        }

    seed = int(cfg.training.seed)
    train_sub = getattr(cfg.dataset.preprocessing, "train_subsample", None)
    val_sub = getattr(cfg.dataset.preprocessing, "val_subsample", None)

    train = raw[str(splits.train)]
    val = raw[str(splits.validation)]
    test = raw[str(splits.test)] if str(splits.test) in raw else raw[str(splits.validation)]

    train = _subsample(train, train_sub, seed=seed)
    val = _subsample(val, val_sub, seed=seed)

    remove_cols = list(train.column_names)
    train = train.map(tok_fn, remove_columns=remove_cols, desc="tokenize-train")
    val = val.map(tok_fn, remove_columns=list(val.column_names), desc="tokenize-val")
    test = test.map(tok_fn, remove_columns=list(test.column_names), desc="tokenize-test")

    train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    return {"train": train, "validation": val, "test": test}


class SummarizationCollator:
    def __init__(self, tokenizer, label_pad_token_id: int = -100):
        self.tokenizer = tokenizer
        self.label_pad_token_id = int(label_pad_token_id)

    def __call__(self, features):
        # already padded to max_length in preprocess
        input_ids = torch.stack([f["input_ids"] for f in features], dim=0)
        attention_mask = torch.stack([f["attention_mask"] for f in features], dim=0)
        labels = torch.stack([f["labels"] for f in features], dim=0)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
