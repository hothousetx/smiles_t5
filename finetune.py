import argparse
import random
from pathlib import Path
import numpy as np
import torch
import transformers
from rdkit import RDLogger
import json
import smiles_t5

RDLogger.DisableLog("rdApp.*")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="the path to the dataset that will be used to train the model",
    )
    parser.add_argument(
        "--val_dataset",
        type=Path,
        required=False,
        help="optional path to the validation datatset, if not given, the dataset will be split according to --split_method",
    )
    parser.add_argument(
        "--test_dataset",
        type=Path,
        required=False,
        help="optional path to the test datatset, if not given, the dataset will be split according to --split_method",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="whether or not to clean the SMILES strings before training. This will remove salts and canonicalise them."
    )
    parser.add_argument(
        "--pretrained_model",
        default="hhtx/smiles_t5",
        required=True,
        type=str,
        help="the name of the model to load",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="the path that you want to save the new model & checkpoints in",
    )
    parser.add_argument(
        "--split_method",
        type=str,
        default="scaffold",
        choices=["random", "scaffold"],
        help="how to split the data if --val_dataset and --test_dataset not given",
    )
    parser.add_argument(
        "--fit_metric",
        type=str,
        default="loss",
        help="the metric that the early stopping callback will monitor to stop training early",
    )
    parser.add_argument(
        "--greater_is_better",
        action="store_true",
        help="whether a higher or lower number is desired from the val_metric",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help="the learning rate to apply to the optimizer",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of epochs to train the model for",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="number of epochs to monitor early stopping for",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-2,
        help="the weight decay to apply to the optimizer",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="indicate the batch size per device (for both train & eval)",
    )
    parser.add_argument(
        "--smiles_col",
        type=str,
        default="smiles",
        help="column in the dataset that contains the SMILES strings",
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default="labels",
        help="the column of the csv that contains the labels",
    )
    parser.add_argument(
        "--dataset_col",
        type=str,
        help="the column to use to split the dataset into train/val/test, labels in this column should be integers, e.g. 0, 1, 2 for train, val, test. \
            If not given, will use the splitting method given by --split_method.",
    )
    parser.add_argument(
        "--tf32",
        action="store_true",
        help="whether to use tf32 during finetuning",
    )
    parser.add_argument(
        "--eval_metrics",
        nargs='+',
        # action='store',
        default=["roc_auc", "prc_auc", "f1", "precision", "recall", "accuracy", "hamming_accuracy"],
        help="the metrics to report during finetuning",
    )
    parser.add_argument(
        "--eval_acc_steps",
        type=int,
        default=None,
        help="the amount of steps of eval before moving data to CPU",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="the random seed to use"
    )

    args = vars(parser.parse_args())
    print(args)


    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        n_devices = torch.cuda.device_count()
        print(f"Found {n_devices} CUDA devices.")

    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    random.seed(args["seed"])

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args["pretrained_model"]
    )

    dataset = smiles_t5.dataset.load_dataset(
        dataset_path=args['dataset'],
        val_path=args['val_dataset'],
        test_path=args['test_dataset'],
        smiles_col=args['smiles_col'],
        splitting_method=args['split_method'],
        clean=args["clean"],
    )

    unique_labels = []
    for labels in np.unique(dataset["train"][args["label_col"]]):
        for label in labels.split(" "):
            if label:
                if label not in unique_labels:
                        unique_labels.append(label)
    tokenizer.add_tokens(unique_labels)

    def tokenize(data):
        results = tokenizer(data[args["smiles_col"]])
        results["labels"] = tokenizer(data[args["label_col"]])["input_ids"]
        return results

    tokenized = dataset.map(
        tokenize,
        batched=True,
        num_proc=1,
        remove_columns=dataset["train"].column_names,
    )


    model = transformers.T5ForConditionalGeneration.from_pretrained(
        args["pretrained_model"],
    )
    model.config.labels = unique_labels
    model.resize_token_embeddings(
        len(tokenizer),
        pad_to_multiple_of=32 if args["tf32"] else None,
    )

    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=tokenizer.pad_token_id,
        pad_to_multiple_of=32 if args["tf32"] else None,
    )

    calc_metrics = smiles_t5.metrics.T5Seq2SeqMetricCalculator(
        tokenizer=tokenizer,
        labels=unique_labels,
        metrics=args["eval_metrics"],
    )

    training_args = transformers.Seq2SeqTrainingArguments(
        output_dir=args["output_dir"],
        evaluation_strategy="epoch",
        learning_rate=args["learning_rate"],
        lr_scheduler_type="linear",
        num_train_epochs=args["epochs"],
        weight_decay=args["weight_decay"],
        per_device_train_batch_size=args["batch_size"],
        per_device_eval_batch_size=args["batch_size"],
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        resume_from_checkpoint=False,
        load_best_model_at_end=True,
        metric_for_best_model=args["fit_metric"],
        greater_is_better=args["greater_is_better"],
        tf32=args['tf32'],
        eval_accumulation_steps=args["eval_acc_steps"],
        seed=args["seed"],
        data_seed=args["seed"],
    )
  
    trainer = transformers.Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["val"],
        data_collator=data_collator,
        compute_metrics=calc_metrics,
        callbacks=[
            transformers.EarlyStoppingCallback(
                early_stopping_patience=args["patience"],
                early_stopping_threshold=1e-6,
            )
        ],
    )

    trainer.train()
    # save model & tokenizer
    trainer.model.save_pretrained(args["output_dir"])
    tokenizer.save_pretrained(args["output_dir"])
    # get test data
    predictions = trainer.predict(tokenized["test"])
    print(f"Test Metric: {predictions.metrics}")
    prediction_exporter = smiles_t5.metrics.PredictionExporter(
        tokenizer=tokenizer,
        labels=model.config.labels,
        path=args["output_dir"],
    )
    prediction_exporter.export_test_predictions(
        predictions.predictions[0],
        predictions.label_ids,
        tokenized["test"],
    )
    prediction_exporter.export_test_metrics(
        predictions.metrics,
    )


if __name__ == "__main__":
    main()
