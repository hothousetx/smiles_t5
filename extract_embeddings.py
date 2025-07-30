import transformers
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
import smiles_t5
import torch


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pretrained_model",
        default="hothousetx/smiles_t5",
        required=True,
        type=str,
        help="the name of the model to load",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="the path to the dataset that will be used",
    )
    parser.add_argument(
        "--smiles_col",
        type=str,
        default="smiles",
        help="column in the dataset that contains the SMILES strings",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="batch size for inference",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="whether to canonicalize the molecules before generating predictions",
    )

    args = parser.parse_args()
    print(args)

    model = transformers.T5EncoderModel.from_pretrained(args.pretrained_model)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.pretrained_model)

    # might need to write a new pipeline
    feature_extractor = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        task='feature-extraction',
        device=0 if torch.cuda.is_available() else -1,
        return_tensors='pt',
    )

    df = pd.read_csv(args.dataset)
    dataset = smiles_t5.dataset.T5InferenceDataset(
        df[args.smiles_col].to_list(),
        clean=args.clean,
    )

    features = []
    for feats in tqdm(feature_extractor(dataset, batch_size=args.batch_size), desc="Extracting embeddings", total=len(dataset)):
        features.append(feats.mean(dim=1))

    # export
    export_dict = dict(zip(dataset.smiles, features))
    torch.save(export_dict, Path(args.dataset.parent, f"{args.dataset.stem}.pt"))
    

if __name__ == "__main__":
    main()
