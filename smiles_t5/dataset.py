import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from datasets import DatasetDict, Dataset, load_from_disk

SALT_REMOVER = SaltRemover()

class T5InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, smiles: List[str], clean: bool = False):
        super(T5InferenceDataset, self).__init__()
        self.smiles = smiles
        self.clean = clean
        self.clean_smiles = clean_smiles
        
    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, i: int):
        if self.clean:
            return clean_smiles(self.smiles[i])
        return self.smiles[i]
    

def clean_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mol, _ = SALT_REMOVER.StripMolWithDeleted(mol)
        smi = Chem.MolToSmiles(mol, canonical=True)
        if smi:
            return smi
    # if not returning then arrive here for the error
    raise TypeError(f"{smiles} is not a valid SMILES string, it doesn't produce a valid mol.")
    

def scaffold_split_dataset(dataset: Dataset, smiles_col: str, fractions: Tuple[float] =(0.8, 0.1, 0.1)):
    rng = np.random.default_rng(0)
    fractions_lens = []
    for f in fractions:
        fractions_lens.append(np.floor(len(dataset) * f))

    def generate_scaffolds(data):
        data['scaffolds'] = [
            MurckoScaffold.MurckoScaffoldSmilesFromSmiles(s, includeChirality=True) 
            for s in data[smiles_col]
        ]
        return data
    
    dataset = dataset.map(generate_scaffolds, batched=True, num_proc=1)
    scaffolds_dict = defaultdict(list)
    for idx, scaff in enumerate(dataset['scaffolds']):
        scaffolds_dict[scaff].append(idx)

    scaffolds = list(scaffolds_dict.values())
    rng.shuffle(scaffolds)

    k_bins = len(fractions_lens)
    bin_idx = defaultdict(list)
    for scaff in scaffolds:
        for k in range(k_bins):
            bin_size = fractions_lens[k]
            if len(bin_idx[k])+len(scaff) < bin_size:
                bin_idx[k].extend(scaff)
                break

    split_dataset = DatasetDict({
        'train': dataset.select(k_bins[0]),
        'val': dataset.select(k_bins[1]),
        'test': dataset.select(k_bins[2])
    })
    return split_dataset

def load_dataset(dataset_path: Path, val_path: Path, test_path: Path, smiles_col: str, splitting_method: str, clean=False):
    if dataset_path.is_dir():
        dataset = load_from_disk(dataset_path)
    elif dataset_path.suffix == ".csv":
        if val_path and test_path:
            train_df = pd.read_csv(dataset_path)
            val_df = pd.read_csv(val_path)
            test_df = pd.read_csv(test_path)
            dataset = DatasetDict({
                "train": Dataset.from_pandas(train_df),
                "val": Dataset.from_pandas(val_df),
                "test": Dataset.from_pandas(test_df),
            })
        else:
            df = pd.read_csv(dataset_path)
            dataset = Dataset.from_pandas(df)
    if clean:
        dataset = dataset.map(
            lambda example: {smiles_col: clean_smiles(example[smiles_col])},
            batched=False,
            num_proc=1
        )
    if type(dataset) == Dataset:
        if splitting_method == "scaffold":
            dataset = scaffold_split_dataset(
                dataset=dataset,
                smiles_col=smiles_col,
            )
        else:  # random split
            dataset_train_test = dataset.train_test_split(test_size=0.2)
            dataset_train = dataset_train_test["train"]
            dataset_val_test = dataset_train_test["test"].train_test_split(
                test_size=0.5
            )
            dataset = DatasetDict({
                "train": dataset_train,
                "val": dataset_val_test["train"],
                "test": dataset_val_test["test"],
            })
    return dataset