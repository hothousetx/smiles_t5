"""Embedding extraction script for T5 encoder models.

This script provides a command-line interface for extracting molecular embeddings
from SMILES strings using a pretrained T5 encoder model. The embeddings are
mean-pooled across the sequence dimension and saved as a PyTorch dictionary
mapping SMILES strings to their corresponding embedding tensors.

These embeddings can be used as fixed feature representations for downstream
machine learning tasks without requiring fine-tuning.

Example usage:
    python extract_embeddings.py \\
        --dataset molecules.csv \\
        --smiles_col smiles \\
        --batch_size 32

Output:
    A .pt file containing a dictionary of {smiles: embedding_tensor} pairs.
"""

from pathlib import Path
from typing import Any

import click
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, T5EncoderModel
from transformers.pipelines import pipeline

import smiles_t5


@click.command()
@click.option(
    "--dataset",
    type=Path,
    required=True,
    help="the path to the dataset that will be used",
)
@click.option(
    "--pretrained_model",
    default="hothousetx/smiles_t5",
    type=str,
    help="the name of the model to load",
)
@click.option(
    "--smiles_col",
    type=str,
    default="smiles",
    help="column in the dataset that contains the SMILES strings",
)
@click.option(
    "--batch_size",
    type=int,
    default=1,
    help="batch size for inference",
)
@click.option(
    "--clean",
    is_flag=True,
    help="whether to canonicalize the molecules before generating predictions",
)
def main(
    dataset: Path,
    pretrained_model: str,
    smiles_col: str,
    batch_size: int,
    clean: bool,
) -> None:
    """Extract molecular embeddings from SMILES strings using a T5 encoder.

    This function loads a pretrained T5 encoder model, processes SMILES strings
    through it, and extracts mean-pooled embeddings. The embeddings are saved
    as a PyTorch file in the same directory as the input dataset.

    The embeddings are computed by:
    1. Tokenizing each SMILES string
    2. Passing through the T5 encoder
    3. Mean-pooling across the sequence dimension

    Args:
        dataset: Path to a CSV file containing SMILES strings.
        pretrained_model: Name or path of the pretrained T5 model to use
            for embedding extraction.
        smiles_col: Column name containing SMILES strings in the dataset.
        batch_size: Number of SMILES strings to process in each batch.
        clean: Whether to canonicalize SMILES and remove salts before
            extracting embeddings.

    Output:
        Saves a .pt file in the same directory as the dataset with the same
        stem name. The file contains a dictionary mapping SMILES strings
        to their embedding tensors.

    Example:
        For input 'molecules.csv', outputs 'molecules.pt' containing:
        {'CCO': tensor([...]), 'c1ccccc1': tensor([...]), ...}
    """
    model = T5EncoderModel.from_pretrained(pretrained_model)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    # Create feature extraction pipeline
    feature_extractor: Any = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="feature-extraction",
        device=0 if torch.cuda.is_available() else -1,
        return_tensors="pt",
    )

    df = pd.read_csv(dataset)
    inference_dataset = smiles_t5.dataset.T5InferenceDataset(
        df[smiles_col].to_list(),
        clean=clean,
    )

    # Convert dataset to list of strings for the pipeline
    smiles_list = [inference_dataset[i] for i in range(len(inference_dataset))]

    features = []
    for feats in tqdm(
        feature_extractor(smiles_list, batch_size=batch_size),
        desc="Extracting embeddings",
        total=len(inference_dataset),
    ):
        features.append(feats.mean(dim=1))

    # Export embeddings as a dictionary mapping SMILES to tensors
    export_dict = dict(zip(inference_dataset.smiles, features))
    output_path = Path(dataset.parent, f"{dataset.stem}.pt")
    torch.save(export_dict, output_path)


if __name__ == "__main__":
    main()
