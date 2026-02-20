"""Prediction module for running inference with trained T5 models.

This module provides functionality for generating predictions from SMILES strings
using a trained T5 model, with optional Monte Carlo dropout for uncertainty
estimation. Results are saved to CSV.

Example usage:
    python -m smiles_t5.predict --dataset data.csv --model_dir ./model --mc_dropout_samples 10
"""

import logging
from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import smiles_t5


def enable_dropout(model: torch.nn.Module) -> None:
    """Enable dropout layers during inference for Monte Carlo dropout.

    This function sets all dropout layers in the model to training mode,
    allowing them to remain active during inference. This is used for
    Monte Carlo dropout, which provides uncertainty estimates by running
    multiple forward passes with different dropout masks.

    Args:
        model: A PyTorch model containing dropout layers.
    """
    for module in model.modules():
        if module.__class__.__name__.startswith("Dropout"):
            module.train()


@click.command()
@click.option(
    "--dataset",
    type=Path,
    required=True,
    help="the path to the dataset that will be used",
)
@click.option(
    "--model_dir",
    type=Path,
    required=True,
    help="the path that you want to load the model & tokenizer from",
)
@click.option("--batch_size", type=int, default=1, help="the batch size for inference")
@click.option(
    "--smiles_col",
    type=str,
    default="smiles",
    help="column in the dataset that contains the SMILES strings",
)
@click.option(
    "--clean",
    is_flag=True,
    default=False,
    help="whether to canonicalize the molecules before generating predictions",
)
@click.option(
    "--output_file",
    type=Path,
    help="the CSV file to save the data in, if not given, it will be in the current directory",
)
@click.option(
    "--mc_dropout_samples",
    type=int,
    default=10,
    help="number of Monte Carlo dropout inference passes for confidence estimation (set to 1 to disable)",
)
def main(
    dataset: Path,
    model_dir: Path,
    batch_size: int,
    smiles_col: str,
    clean: bool,
    output_file: Optional[Path] = None,
    mc_dropout_samples: int = 10,
) -> None:
    """Run inference on a dataset using a trained T5 model.

    This function loads a trained model and generates predictions for all SMILES
    strings in the input dataset. When mc_dropout_samples > 1, it uses Monte Carlo
    dropout to estimate prediction uncertainty by running multiple forward passes
    with dropout enabled.

    Output columns (when mc_dropout_samples > 1):
        - {label}_0, {label}_1, ..., {label}_n: Individual MC dropout predictions
        - {label}_mean: Mean prediction across all MC samples
        - {label}_std: Standard deviation (uncertainty) across MC samples
        - confidence: Overall confidence score (1 - mean std across all labels)

    Output columns (when mc_dropout_samples = 1):
        - {label}: Single prediction for each label

    Args:
        dataset: Path to the input CSV file containing SMILES strings.
        model_dir: Path to the directory containing the trained model and tokenizer.
        batch_size: Number of samples to process in each batch.
        smiles_col: Name of the column containing SMILES strings in the dataset.
        clean: Whether to canonicalize SMILES strings before prediction.
        output_file: Path for the output CSV file. If None, saves to the dataset
            directory with '_predictions' suffix.
        mc_dropout_samples: Number of Monte Carlo dropout passes. Set to 1 to
            disable uncertainty estimation.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("inference.log", mode="w"),
        ],
    )
    logging.info("Starting inference...")
    logging.info(f"Loading model from {model_dir}")
    pipeline = smiles_t5.pipeline.T5Seq2SeqPipeline(str(model_dir))

    logging.info(f"Loading dataset from {dataset}")
    if not dataset.exists():
        logging.error(f"Dataset file {dataset} does not exist.")
        raise FileNotFoundError(f"Dataset file {dataset} does not exist.")

    df = pd.read_csv(dataset)
    df.dropna(subset=[smiles_col], inplace=True)

    inf_dataset = smiles_t5.dataset.T5InferenceDataset(
        df[smiles_col].to_list(), clean=clean
    )

    dataloader = DataLoader(inf_dataset, batch_size=batch_size)

    if mc_dropout_samples > 1:
        logging.info(f"Running Monte Carlo dropout with {mc_dropout_samples} samples")
        # Enable dropout for MC dropout inference
        enable_dropout(pipeline.model)

        # Store all predictions across MC samples: shape will be (n_samples, n_datapoints, n_labels)
        all_mc_predictions = []

        for mc_iter in range(mc_dropout_samples):
            predictions_iter = []
            pbar = tqdm(
                dataloader,
                desc=f"MC Dropout pass {mc_iter + 1}/{mc_dropout_samples}",
                total=len(dataloader),
            )

            for smiles in pbar:
                with torch.no_grad():
                    outputs = pipeline(smiles)
                predictions_iter.extend([list(i.values()) for i in outputs])

            all_mc_predictions.append(predictions_iter)

        # Convert to numpy array for easier computation
        # Shape: (mc_samples, n_datapoints, n_labels)
        all_mc_predictions = np.array(all_mc_predictions)

        # Calculate mean predictions across MC samples
        mean_predictions = np.mean(all_mc_predictions, axis=0)

        # Calculate standard deviation (uncertainty) across MC samples
        std_predictions = np.std(all_mc_predictions, axis=0)

        logging.info("MC Dropout predictions generated successfully.")
        logging.info("Adding predictions and confidence metrics to the DataFrame...")

        # Add individual predictions for each MC sample: {label}_0, {label}_1, etc.
        for mc_iter in range(mc_dropout_samples):
            iter_columns = [f"{label}_{mc_iter}" for label in pipeline.labels]
            df[iter_columns] = all_mc_predictions[mc_iter]

        # Add mean predictions: {label}_mean
        mean_columns = [f"{label}_mean" for label in pipeline.labels]
        df[mean_columns] = mean_predictions

        # Add standard deviation (uncertainty) for each label: {label}_std
        std_columns = [f"{label}_std" for label in pipeline.labels]
        df[std_columns] = std_predictions

        # Add overall confidence score (inverse of mean std across all labels)
        mean_std_per_sample = np.mean(std_predictions, axis=1)
        df["confidence"] = 1 - mean_std_per_sample  # Higher confidence = lower std

    else:
        logging.info("Running single inference pass (MC dropout disabled)")
        predictions = []
        pbar = tqdm(dataloader, desc="Generating predictions", total=len(dataloader))

        for smiles in pbar:
            outputs = pipeline(smiles)
            predictions.extend([list(i.values()) for i in outputs])

        logging.info("Predictions generated successfully.")
        logging.info("Adding predictions to the DataFrame...")
        df[pipeline.labels] = predictions

    if output_file is None:
        logging.info("No output file given, saving to the location of the dataset")
        output_file = Path(dataset.parent, f"{dataset.stem}_predictions.csv")
    logging.info(f"Saving predictions to {output_file}")
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
