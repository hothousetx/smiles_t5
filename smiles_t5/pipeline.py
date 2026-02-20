"""Inference pipeline for T5 seq2seq molecular property prediction.

This module provides a pipeline class for running inference with fine-tuned
T5 models on SMILES strings. It handles tokenization, model inference, and
conversion of token probabilities to label scores.

Example:
    >>> from smiles_t5.pipeline import T5Seq2SeqPipeline
    >>> pipeline = T5Seq2SeqPipeline("path/to/finetuned/model")
    >>> predictions = pipeline(["CCO", "c1ccccc1"])
    >>> print(predictions[0])
    {'toxic': 0.12, 'non_toxic': 0.88}
"""

from typing import Dict, List, Optional, Sequence, Union

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.tokenization_utils import PreTrainedTokenizer


class T5Seq2SeqPipeline:
    """Pipeline for running inference with fine-tuned T5 seq2seq models.

    This class provides a simple interface for generating multi-label
    classification predictions from SMILES strings using a fine-tuned T5 model.
    It handles tokenization, generation, and conversion of output scores to
    per-label probabilities.

    The pipeline extracts the maximum softmax probability for each label token
    across all positions in the generated sequence, providing a confidence
    score for each label.

    Attributes:
        tokenizer: The tokenizer for encoding SMILES strings.
        model: The T5 model for conditional generation.
        device: The device (cuda/cpu/mps) where the model is loaded.
        labels: List of label strings from the model config.
        label_ids: Token IDs corresponding to each label.

    Example:
        >>> pipeline = T5Seq2SeqPipeline("path/to/model")
        >>> results = pipeline(["CCO", "c1ccccc1"])
        >>> for smiles, scores in zip(["CCO", "c1ccccc1"], results):
        ...     print(f"{smiles}: {scores}")
    """

    def __init__(
        self,
        pretrained_path: Optional[str] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        model: Optional[T5ForConditionalGeneration] = None,
        device_map: str = "auto",
    ):
        """Initialize the inference pipeline.

        The pipeline can be initialized either from a pretrained model path
        or by providing a tokenizer and model directly.

        Args:
            pretrained_path: Path to a directory containing a fine-tuned model
                and tokenizer. If provided, tokenizer and model args are ignored.
            tokenizer: A pretrained tokenizer. Required if pretrained_path is None.
            model: A T5ForConditionalGeneration model. Required if pretrained_path
                is None.
            device_map: Device placement strategy. Options are:
                - "auto": Use CUDA if available, otherwise CPU
                - "cuda": Force CUDA (will error if unavailable)
                - "cpu": Force CPU
                - "mps": Use Apple Metal Performance Shaders

        Raises:
            ValueError: If neither pretrained_path nor both tokenizer and model
                are provided, or if device_map is invalid.
        """
        if pretrained_path is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
            self.model = T5ForConditionalGeneration.from_pretrained(pretrained_path)
        else:
            if tokenizer is None or model is None:
                raise ValueError(
                    "Either pretrained_path or both tokenizer and model must be provided."
                )
            self.tokenizer = tokenizer
            self.model = model

        if device_map in ["cuda", "cpu", "mps"]:
            self.device = device_map
        elif device_map == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            raise ValueError("device_map must be 'auto', 'cpu', 'mps', or 'cuda'.")
        self.model.eval()
        self.model.to(self.device)  # type: ignore

        self.labels = self.model.config.labels
        self.label_ids = self.tokenizer.convert_tokens_to_ids(self.labels)

    def __call__(
        self, inputs: Union[Sequence[str], Dataset], **generate_kwargs
    ) -> List[Dict[str, float]]:
        """Generate predictions for SMILES strings.

        This method tokenizes the input SMILES strings, runs the T5 model
        in generation mode, and converts the output token probabilities
        to per-label scores.

        Args:
            inputs: SMILES strings to generate predictions for. Can be:
                - A single SMILES string
                - A list of SMILES strings
                - A PyTorch Dataset yielding SMILES strings
            **generate_kwargs: Additional keyword arguments passed to the
                model's generate() method (e.g., num_beams, temperature).

        Returns:
            A list of dictionaries, one per input SMILES. Each dictionary
            maps label names to their predicted probability scores (0-1).

        Raises:
            ValueError: If inputs is not a string, list of strings, or Dataset,
                or if a list contains non-string elements.

        Example:
            >>> pipeline = T5Seq2SeqPipeline("path/to/model")
            >>> results = pipeline(["CCO"])
            >>> print(results[0])
            {'label1': 0.85, 'label2': 0.12}
        """
        if isinstance(inputs, str):
            inputs = [inputs]
        elif isinstance(inputs, Dataset):
            inputs = [i for i in inputs]
        elif isinstance(inputs, list):
            for i in inputs:
                if not isinstance(i, str):
                    raise ValueError(
                        "If inputs is a list, all elements must be strings."
                    )
        else:
            raise ValueError("Inputs must be a string, list of strings, or a Dataset.")

        model_inputs = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        generate_kwargs["min_length"] = 0
        generate_kwargs["max_length"] = (
            len(self.labels) + 1
        )  # +1 to compensate for the pad token
        generate_kwargs["return_dict_in_generate"] = True
        generate_kwargs["output_scores"] = True

        model_outputs = self.model.generate(**model_inputs, **generate_kwargs)
        output_scores = torch.stack(model_outputs.scores, dim=0).permute(1, 0, 2)  # type: ignore

        softmax_scores = output_scores.softmax(dim=-1)
        max_scores = softmax_scores.max(dim=1).values
        label_scores = max_scores[:, self.label_ids]
        scores: List[Dict[str, float]] = []
        for i in range(len(inputs)):
            scores_dict: Dict[str, float] = dict(
                zip(self.labels, label_scores[i].tolist())
            )
            scores.append(scores_dict)
        return scores
