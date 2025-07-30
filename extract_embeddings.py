import transformers
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import smiles_t5
import torch
import click

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
):
    model = transformers.T5EncoderModel.from_pretrained(pretrained_model)
    tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model)

    # might need to write a new pipeline
    feature_extractor = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        task='feature-extraction',
        device=0 if torch.cuda.is_available() else -1,
        return_tensors='pt',
    )

    df = pd.read_csv(dataset)
    inference_dataset = smiles_t5.dataset.T5InferenceDataset(
        df[smiles_col].to_list(),
        clean=clean,
    )

    features = []
    for feats in tqdm(feature_extractor(inference_dataset, batch_size=batch_size), desc="Extracting embeddings", total=len(inference_dataset)):
        features.append(feats.mean(dim=1))

    # export
    export_dict = dict(zip(inference_dataset.smiles, features))
    torch.save(export_dict, Path(dataset.parent, f"{dataset.stem}.pt"))
    

if __name__ == "__main__":
    main()
