"""SmilesT5: A package for molecular property prediction using T5 transformers.

This package provides tools for fine-tuning and inference with T5-based models
on molecular property prediction tasks using SMILES representations.

Modules:
    dataset: Data loading, preprocessing, and splitting utilities for molecular datasets.
    pipeline: Inference pipeline for generating predictions from SMILES strings.
    metrics: Evaluation metrics and prediction export utilities for seq2seq models.

Example:
    >>> import smiles_t5
    >>> pipeline = smiles_t5.pipeline.T5Seq2SeqPipeline("path/to/model")
    >>> predictions = pipeline(["CCO", "c1ccccc1"])
"""

from . import dataset, metrics, pipeline

__all__ = ["dataset", "metrics", "pipeline"]
