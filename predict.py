import transformers
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
import smiles_t5
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="the path to the dataset that will be used",
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        required=True,
        help="the path that you want to load the model & tokenizer from",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="the batch size for inference"
    )
    parser.add_argument(
        "--smiles_col",
        type=str,
        default="smiles",
        help="column in the dataset that contains the SMILES strings",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="whether to canonicalize the molecules before generating predictions",
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        help="the CSV file to save the data in, if not given, it will be in the current directory",
    )

    args = parser.parse_args()
    print(args)

    # model = transformers.T5ForConditionalGeneration.from_pretrained(args.model_dir)
    # tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_dir)

    pipeline = smiles_t5.pipeline.T5Seq2SeqPipeline(args.model_dir)

    df = pd.read_csv(args.dataset)
    df.dropna(subset=[args.smiles_col], inplace=True)

    dataset = smiles_t5.dataset.T5InferenceDataset(
        df[args.smiles_col].to_list(), clean=args.clean
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    predictions = []
    pbar = tqdm(dataloader, desc="Generating predictions", total=len(dataloader))

    for smiles in pbar:
        outputs = pipeline(smiles)
        predictions.extend([list(i.values()) for i in outputs])

    df[pipeline.labels] = predictions

    if not args.output_file:
        args.output_file = Path(args.dataset.parent, f"{args.dataset.stem}_predictions.csv")
    df.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()
