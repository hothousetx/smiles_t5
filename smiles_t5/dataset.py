"""Dataset utilities for loading, preprocessing, and splitting molecular datasets.

This module provides functions and classes for working with molecular datasets
in the context of T5-based property prediction. It supports loading data from
CSV files or HuggingFace datasets, cleaning SMILES strings, and splitting
datasets using random or scaffold-based methods.

Example:
    >>> from smiles_t5.dataset import load_dataset, T5InferenceDataset
    >>> dataset = load_dataset(Path("data.csv"), smiles_col="smiles", splitting_method="scaffold")
    >>> inference_data = T5InferenceDataset(["CCO", "c1ccccc1"], clean=True)
"""

import logging
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.Scaffolds import MurckoScaffold

SALT_REMOVER = SaltRemover()

logger = logging.getLogger(__name__)


class T5InferenceDataset(torch.utils.data.Dataset):
    """A PyTorch Dataset for SMILES strings used during inference.

    This dataset wraps a list of SMILES strings and optionally applies
    canonicalization and salt removal during item retrieval.

    Attributes:
        smiles: List of SMILES strings.
        clean: Whether to clean SMILES strings when accessed.

    Example:
        >>> dataset = T5InferenceDataset(["CCO", "CC(=O)O"], clean=True)
        >>> len(dataset)
        2
        >>> dataset[0]
        'CCO'
    """

    def __init__(self, smiles: List[str], clean: bool = False):
        """Initialize the inference dataset.

        Args:
            smiles: List of SMILES strings to use for inference.
            clean: Whether to canonicalize and remove salts from SMILES
                strings when accessed. Defaults to False.
        """
        super(T5InferenceDataset, self).__init__()
        self.smiles = smiles
        self.clean = clean
        self.clean_smiles = clean_smiles

    def __len__(self) -> int:
        """Return the number of SMILES strings in the dataset."""
        return len(self.smiles)

    def __getitem__(self, i: int) -> str:
        """Get a SMILES string by index.

        Args:
            i: Index of the SMILES string to retrieve.

        Returns:
            The SMILES string at the given index, optionally cleaned.
        """
        if self.clean:
            return clean_smiles(self.smiles[i])
        return self.smiles[i]


def clean_smiles(smiles: str) -> str:
    """Clean a SMILES string by removing salts and canonicalizing.

    This function parses the SMILES string, removes any salt fragments,
    and returns the canonical SMILES representation. If the input is
    invalid, a warning is logged and the original string is returned.

    Args:
        smiles: A SMILES string to clean.

    Returns:
        The cleaned, canonical SMILES string, or the original string
        if parsing failed.

    Example:
        >>> clean_smiles("CCO.Cl")  # Removes salt
        'CCO'
        >>> clean_smiles("C(C)O")  # Canonicalizes
        'CCO'
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mol, _ = SALT_REMOVER.StripMolWithDeleted(mol)
        smi = Chem.MolToSmiles(mol, canonical=True)
        if smi:
            return smi
    # if not returning then arrive here for the error
    logger.warning(
        f"{smiles} is not a valid SMILES string, it doesn't produce a valid mol. Returning the original SMILES."
    )
    return smiles


def generate_scaffolds(data: dict, smiles_col: str) -> dict:
    """Generate Murcko scaffolds for a batch of molecules.

    This function extracts the Murcko scaffold (core ring structure) from
    each molecule in the batch. Scaffolds are used for splitting datasets
    to ensure structural diversity between train/val/test sets.

    Args:
        data: A dictionary containing a column of SMILES strings.
        smiles_col: The key in the data dictionary containing SMILES strings.

    Returns:
        The input dictionary with an added 'scaffolds' key containing
        the Murcko scaffold SMILES for each molecule.
    """
    data["scaffolds"] = [
        MurckoScaffold.MurckoScaffoldSmilesFromSmiles(s, includeChirality=True)
        for s in data[smiles_col]
    ]
    return data


def scaffold_split_dataset(
    dataset: Dataset,
    smiles_col: str,
    fractions: Tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> DatasetDict:
    """Split a dataset based on molecular scaffolds.

    This function groups molecules by their Murcko scaffolds and assigns
    entire scaffold groups to train/val/test splits. This ensures that
    molecules with similar core structures are not split across sets,
    providing a more realistic evaluation of model generalization.

    Args:
        dataset: A HuggingFace Dataset containing SMILES strings.
        smiles_col: The column name containing SMILES strings.
        fractions: A tuple of (train, val, test) fractions. Must sum to 1.0.
            Defaults to (0.8, 0.1, 0.1).

    Returns:
        A DatasetDict with 'train', 'val', and 'test' splits.

    Note:
        The actual split sizes may differ slightly from the requested
        fractions because entire scaffold groups are kept together.
    """
    rng = np.random.default_rng(0)
    fractions_lens = []
    for f in fractions:
        fractions_lens.append(np.floor(len(dataset) * f))

    logger.info("Generating scaffolds for the dataset")
    generate_scaffs = partial(generate_scaffolds, smiles_col=smiles_col)
    dataset = dataset.map(generate_scaffs, batched=True, num_proc=1)
    scaffolds_dict = defaultdict(list)
    for idx, scaff in enumerate(dataset["scaffolds"]):
        scaffolds_dict[scaff].append(idx)

    logger.info(f"Found {len(scaffolds_dict)} unique scaffolds in the dataset")
    scaffolds = list(scaffolds_dict.values())
    logger.info("Shuffling scaffolds to ensure randomness")
    rng.shuffle(scaffolds)

    logger.info("Splitting scaffolds into train, val, and test sets")
    k_bins = len(fractions_lens)
    bin_idx = defaultdict(list)
    for scaff in scaffolds:
        for k in range(k_bins):
            bin_size = fractions_lens[k]
            if len(bin_idx[k]) + len(scaff) < bin_size:
                bin_idx[k].extend(scaff)
                break

    split_dataset = DatasetDict(
        {
            "train": dataset.select(bin_idx[0]),
            "val": dataset.select(bin_idx[1]),
            "test": dataset.select(bin_idx[2]),
        }
    )
    return split_dataset


def load_dataset(
    dataset_path: Path,
    smiles_col: str,
    splitting_method: str,
    clean: bool = False,
    test_path: Optional[Path] = None,
    val_path: Optional[Path] = None,
) -> DatasetDict:
    """Load and optionally split a molecular dataset.

    This function loads a dataset from a CSV file or HuggingFace dataset
    directory, optionally cleans the SMILES strings, and splits the data
    into train/val/test sets if not already split.

    Args:
        dataset_path: Path to a CSV file or HuggingFace dataset directory.
        smiles_col: The column name containing SMILES strings.
        splitting_method: Method for splitting data if val_path and test_path
            are not provided. Options are 'scaffold' or 'random'.
        clean: Whether to canonicalize and remove salts from SMILES strings.
            Defaults to False.
        test_path: Optional path to a separate test CSV file. If provided
            along with val_path, no automatic splitting is performed.
        val_path: Optional path to a separate validation CSV file. If provided
            along with test_path, no automatic splitting is performed.

    Returns:
        A DatasetDict with 'train', 'val', and 'test' splits.

    Example:
        >>> dataset = load_dataset(
        ...     Path("molecules.csv"),
        ...     smiles_col="smiles",
        ...     splitting_method="scaffold",
        ...     clean=True
        ... )
        >>> len(dataset["train"])
        800
    """
    logger.info(f"Loading dataset from {dataset_path}")
    dataset: Dataset | DatasetDict
    if dataset_path.is_dir():
        logger.info("Loading dataset from disk: {dataset_path}")
        dataset = load_from_disk(dataset_path)
    elif dataset_path.suffix == ".csv":
        logger.info("Loading dataset from CSV file")
        if val_path and test_path:
            logger.info("Loading dataset with train, val, and test splits")
            train_df = pd.read_csv(dataset_path)
            val_df = pd.read_csv(val_path)
            test_df = pd.read_csv(test_path)
            dataset = DatasetDict(
                {
                    "train": Dataset.from_pandas(train_df),
                    "val": Dataset.from_pandas(val_df),
                    "test": Dataset.from_pandas(test_df),
                }
            )
        else:
            logger.info("Loading dataset with no splits, using the whole dataset")
            df = pd.read_csv(dataset_path)
            dataset = Dataset.from_pandas(df)
    else:
        raise ValueError(
            f"Unsupported dataset format: {dataset_path}. "
            "Expected a directory (HuggingFace dataset) or a .csv file."
        )
    if clean:
        logger.info("Cleaning SMILES strings in the dataset")
        dataset = dataset.map(
            lambda example: {smiles_col: clean_smiles(example[smiles_col])},
            batched=False,
            num_proc=1,
        )
    if type(dataset) is Dataset:
        if splitting_method == "scaffold":
            logger.info("Splitting dataset using scaffold method")
            dataset = scaffold_split_dataset(
                dataset=dataset,
                smiles_col=smiles_col,
            )
        else:  # random split
            logger.info("Splitting dataset randomly into train, val, and test sets")
            dataset_train_test = dataset.train_test_split(test_size=0.2)
            dataset_train = dataset_train_test["train"]
            dataset_val_test = dataset_train_test["test"].train_test_split(
                test_size=0.5
            )
            dataset = DatasetDict(
                {
                    "train": dataset_train,
                    "val": dataset_val_test["train"],
                    "test": dataset_val_test["test"],
                }
            )
    assert isinstance(dataset, DatasetDict), (
        "Dataset should be a DatasetDict at this point"
    )
    return dataset
