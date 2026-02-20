"""Metrics and evaluation utilities for T5 seq2seq molecular property prediction.

This module provides classes for computing evaluation metrics during training
and for exporting predictions and metrics after model evaluation. It handles
the conversion between T5's token-based outputs and multi-label classification
metrics.

Example:
    >>> from smiles_t5.metrics import T5Seq2SeqMetricCalculator, PredictionExporter
    >>> calculator = T5Seq2SeqMetricCalculator(tokenizer, labels, metrics=["roc_auc", "f1"])
    >>> exporter = PredictionExporter(tokenizer, labels, Path("./output"))
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from numpy.typing import NDArray
from torcheval.metrics import functional as metrics
from transformers.tokenization_utils import PreTrainedTokenizer


class T5Seq2SeqMetricCalculator:
    """Calculator for evaluation metrics in T5 seq2seq multi-label classification.

    This class computes various classification metrics by converting T5's
    token-level outputs into multi-label predictions. It extracts the maximum
    probability for each label token across all sequence positions and compares
    against binary ground truth labels.

    Supported metrics:
        - roc_auc: Area under the ROC curve (per-label, then averaged)
        - prc_auc: Area under the Precision-Recall curve
        - f1: F1 score at threshold 0.5
        - precision: Precision at threshold 0.5
        - recall: Recall at threshold 0.5
        - accuracy: Exact match accuracy across all labels
        - hamming_accuracy: Hamming accuracy (per-label accuracy averaged)

    Attributes:
        tokenizer: The tokenizer used for encoding/decoding.
        labels: List of label token strings.
        label_token_ids: Token IDs corresponding to each label.
        n_labels: Number of labels.
        metrics: List of metric names to compute.

    Example:
        >>> calculator = T5Seq2SeqMetricCalculator(
        ...     tokenizer=tokenizer,
        ...     labels=["toxic", "severe_toxic", "obscene"],
        ...     metrics=["roc_auc", "f1"]
        ... )
        >>> results = calculator((logits, labels))
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        labels: List[str],
        metrics: List[str] = ["roc_auc"],
    ):
        """Initialize the metric calculator.

        Args:
            tokenizer: A pretrained tokenizer that includes the label tokens.
            labels: List of label strings that were added to the tokenizer's
                vocabulary during fine-tuning.
            metrics: List of metric names to compute. Defaults to ["roc_auc"].
                See class docstring for supported metrics.
        """
        self.tokenizer = tokenizer
        self.labels = labels
        self.label_token_ids = self.tokenizer.convert_tokens_to_ids(self.labels)
        self.n_labels = len(self.labels)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.metrics = metrics

    def __call__(self, data: Tuple[NDArray, NDArray]) -> Dict[str, float]:
        """Compute metrics from model predictions and labels.

        Args:
            data: A tuple of (logits, labels) where:
                - logits: Model output logits of shape
                  (batch_size, sequence_length, vocab_size)
                - labels: Ground truth token IDs of shape
                  (batch_size, max_label_tokens)

        Returns:
            A dictionary mapping metric names to their computed values.
        """
        # make results dict
        results: Dict[str, float] = {}
        np_logits: NDArray[np.float32] = data[0]
        np_labels: NDArray[np.int32] = data[1]
        # logits are (generated_tokens, last_hidden_state)
        # take just the generated tokens
        # shape = (n_batch, sequence_length, vocab_size)
        logits = torch.tensor(np_logits[0])
        # labels are (n_batch, n_tokens + </s>)
        labels = torch.tensor(np_labels)
        if -100 in labels:
            # this shouldn't be necessary but DataCollatorForSeq2Seq doesn't use label_pad_token_id properly
            # replace -100 with the pad token id
            labels[labels == -100] = int(self.tokenizer.pad_token_id)  # type: ignore
        # get the number of labels OR outputted sequence length, which ever is lowest
        min_labels = min(
            self.n_labels, logits.shape[1] - 1
        )  # shape-1 as EOS shouldn't count
        # softmax over all tokens
        probas = logits.softmax(dim=-1)
        # get the logits for the full sequence (excl. EOS token) and only the label tokens
        probas = probas[:, :min_labels, self.label_token_ids]
        # get the max for each label token
        probas_max = probas.max(dim=1).values
        # LABELS make empty binary matrix of all tokens in the vocab
        binary_labels = torch.zeros(labels.shape[0], len(self.tokenizer))
        # scatter 1 values into where the labels are given
        # https://stackoverflow.com/questions/68274722/why-does-torch-scatter-requires-a-smaller-shape-for-indices-than-values
        # binary_labels.scatter_(1, labels, torch.ones_like(binary_labels))
        rows = torch.arange(0, binary_labels.shape[0])[:, None]
        binary_labels[rows.repeat(1, labels.shape[1]), labels] = 1
        # make a matrix of just the possible predicted labels (n_batch, n_labels)
        binary_labels = binary_labels[:, self.label_token_ids]
        for metric in self.metrics:
            if metric == "roc_auc":
                # have to transpose probas and labels as each task is taken independently, then mean across them
                results["roc_auc"] = (
                    torch.stack(
                        [
                            metrics.binary_auroc(probas_max[:, i], binary_labels[:, i])
                            for i in range(self.n_labels)
                        ]
                    )
                    .mean()
                    .item()
                )
            elif metric == "prc_auc":
                results["prc_auc"] = (
                    torch.stack(
                        [
                            metrics.binary_auprc(probas_max[:, i], binary_labels[:, i])
                            for i in range(self.n_labels)
                        ]
                    )
                    .mean()
                    .item()
                )
            elif metric == "f1":
                results["f1"] = (
                    torch.stack(
                        [
                            metrics.binary_f1_score(
                                probas_max[:, i], binary_labels[:, i], threshold=0.5
                            )
                            for i in range(self.n_labels)
                        ]
                    )
                    .mean()
                    .item()
                )
            elif metric == "precision":
                results["precision"] = (
                    torch.stack(
                        [
                            metrics.binary_precision(
                                probas_max[:, i].round().int(),
                                binary_labels[:, i].int(),
                                threshold=0.5,
                            )
                            for i in range(self.n_labels)
                        ]
                    )
                    .mean()
                    .item()
                )
            elif metric == "recall":
                results["recall"] = (
                    torch.stack(
                        [
                            metrics.binary_recall(
                                probas_max[:, i].round().int(),
                                binary_labels[:, i].int(),
                                threshold=0.5,
                            )
                            for i in range(self.n_labels)
                        ]
                    )
                    .mean()
                    .item()
                )
            elif metric == "accuracy":
                results["accuracy"] = metrics.multilabel_accuracy(
                    probas_max, binary_labels, threshold=0.5, criteria="exact_match"
                ).item()
            elif metric == "hamming_accuracy":
                results["hamming_accuracy"] = metrics.multilabel_accuracy(
                    probas_max, binary_labels, threshold=0.5, criteria="hamming"
                ).item()
        return results


class PredictionExporter:
    """Exporter for saving model predictions and metrics to files.

    This class handles the conversion of T5 model outputs to human-readable
    formats and saves them to CSV (predictions) and JSON (metrics) files.

    Attributes:
        tokenizer: The tokenizer used for decoding.
        labels: List of label token strings.
        label_token_ids: Token IDs corresponding to each label.
        n_labels: Number of labels.
        path: Output directory path.

    Example:
        >>> exporter = PredictionExporter(tokenizer, labels, Path("./output"))
        >>> exporter.export_test_predictions(logits, labels, test_dataset)
        >>> exporter.export_test_metrics({"roc_auc": 0.85, "f1": 0.78})
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, labels: List[str], path: Path):
        """Initialize the prediction exporter.

        Args:
            tokenizer: A pretrained tokenizer that includes the label tokens.
            labels: List of label strings used by the model.
            path: Directory path where output files will be saved.
        """
        self.tokenizer = tokenizer
        self.labels = labels
        self.label_token_ids = self.tokenizer.convert_tokens_to_ids(self.labels)
        self.n_labels = len(self.labels)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.path = path

    def export_test_predictions(
        self,
        np_logits: NDArray[np.float32],
        np_labels: NDArray[np.float32],
        test_dataset: Dataset,
    ) -> None:
        """Export test predictions to a CSV file.

        This method converts model logits to probabilities, extracts the
        maximum probability for each label, and saves them alongside the
        binary ground truth labels.

        Args:
            np_logits: Model output logits of shape
                (batch_size, sequence_length, vocab_size).
            np_labels: Ground truth token IDs of shape
                (batch_size, max_label_tokens).
            test_dataset: The tokenized test dataset containing input_ids
                for decoding SMILES strings.

        Output:
            Creates 'test_predictions.csv' in self.path with columns:
                - smiles: Decoded SMILES strings
                - {label}_probas: Predicted probability for each label
                - {label}_label: Binary ground truth for each label
        """
        logits = torch.tensor(np_logits)
        # PROBAS softmax the logits to get probability of all tokens
        probas = logits.softmax(dim=-1)
        # get the max probability of each token regardless of position in the prediction
        probas_max = probas.max(dim=1).values
        # get the max probability of just the defined label tokens (n_batch, n_labels)
        probas_max = probas_max[:, self.label_token_ids]
        # LABELS
        labels = torch.tensor(np_labels)
        if -100 in labels:
            # this shouldn't be necessary but DataCollatorForSeq2Seq doesn't use label_pad_token_id properly
            # replace -100 with the pad token id
            labels[labels == -100] = int(self.tokenizer.pad_token_id)  # type: ignore
        # make empty binary matrix of all tokens in the vocab
        binary_labels = torch.zeros(labels.shape[0], len(self.tokenizer))
        # scatter 1 values into where the labels are given
        binary_labels.scatter_(1, labels, torch.ones_like(binary_labels))
        # make a matrix of just the possible predicted labels (n_batch, n_labels)
        binary_labels = binary_labels[:, self.label_token_ids]

        decoded = [
            d.replace(" ", "")
            for d in self.tokenizer.batch_decode(
                test_dataset["input_ids"], skip_special_tokens=True
            )
        ]
        data: Dict[str, Any] = dict(smiles=decoded)
        for idx, label in enumerate(self.labels):
            data[f"{label}_probas"] = probas_max[:, idx].numpy()
            data[f"{label}_label"] = binary_labels[:, idx].numpy()

        df = pd.DataFrame.from_dict(data)
        df.to_csv(Path(self.path, "test_predictions.csv"), index=False)

    def export_test_metrics(self, test_metrics: Dict[str, float]) -> None:
        """Export test metrics to a JSON file.

        Args:
            test_metrics: Dictionary mapping metric names to their values.

        Output:
            Creates 'test_metrics.json' in self.path containing the metrics.
        """
        json.dump(
            test_metrics,
            Path(self.path, "test_metrics.json").open(mode="w"),
            indent=True,
        )
