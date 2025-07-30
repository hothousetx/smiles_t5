import pandas as pd
from pathlib import Path
from tqdm import tqdm
import smiles_t5
from torch.utils.data import DataLoader
import click


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
@click.option(
    "--batch_size",
    type=int,
    default=1,
    help="the batch size for inference"
)
@click.option(
    "--smiles_col",
    type=str,
    default="smiles",
    help="column in the dataset that contains the SMILES strings",
)
@click.option(
    "--clean",
    is_flag=True,
    help="whether to canonicalize the molecules before generating predictions",
)
@click.option(
    "--output_file",
    type=Path,
    help="the CSV file to save the data in, if not given, it will be in the current directory",
)
def main(
    dataset: Path,
    model_dir: Path,
    batch_size: int,
    smiles_col: str,
    clean: bool,
    output_file: Path,
):

    pipeline = smiles_t5.pipeline.T5Seq2SeqPipeline(model_dir)

    df = pd.read_csv(dataset)
    df.dropna(subset=[smiles_col], inplace=True)

    dataset = smiles_t5.dataset.T5InferenceDataset(
        df[smiles_col].to_list(), clean=clean
    )

    dataloader = DataLoader(dataset, batch_size=batch_size)

    predictions = []
    pbar = tqdm(dataloader, desc="Generating predictions", total=len(dataloader))

    for smiles in pbar:
        outputs = pipeline(smiles)
        predictions.extend([list(i.values()) for i in outputs])

    df[pipeline.labels] = predictions

    if not output_file:
        output_file = Path(dataset.parent, f"{dataset.stem}_predictions.csv")
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
