"""Fine-tuning script for T5 models on molecular property prediction tasks.

This script provides a command-line interface for fine-tuning pretrained T5 models
(e.g., SmilesT5) on multi-label classification tasks using SMILES representations.
It supports scaffold and random data splitting, early stopping, and various
evaluation metrics.

The fine-tuned model and tokenizer are saved to the output directory along with
test predictions and metrics.

Example usage:
    python finetune.py \\
        --dataset molecules.csv \\
        --output_dir ./output \\
        --split_method scaffold \\
        --label_col labels \\
        --eval_metrics roc_auc f1 \\
        --fit_metric roc_auc \\
        --greater_is_better
"""

import random
from pathlib import Path
from typing import List, cast

import click
import numpy as np
import torch
from rdkit import RDLogger
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.data.data_collator import DataCollatorForSeq2Seq
from transformers.trainer_callback import EarlyStoppingCallback
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments

import smiles_t5

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]


@click.command()
@click.option(
    "--dataset",
    type=Path,
    required=True,
    envvar="DATASET",
    help="the path to the dataset that will be used to train the model",
)
@click.option(
    "--val_dataset",
    type=Path,
    default=None,
    envvar="VAL_DATASET",
    help="optional path to the validation datatset, if not given, the dataset will be split according to --split_method",
)
@click.option(
    "--test_dataset",
    type=Path,
    default=None,
    envvar="TEST_DATASET",
    help="optional path to the test datatset, if not given, the dataset will be split according to --split_method",
)
@click.option(
    "--clean",
    is_flag=True,
    help="whether or not to clean the SMILES strings before training. This will remove salts and canonicalise them.",
)
@click.option(
    "--pretrained_model",
    default="hothousetx/smiles_t5",
    type=str,
    envvar="PRETRAINED_MODEL",
    help="the name of the model on huggingface or path to the model to load",
)
@click.option(
    "--output_dir",
    type=Path,
    default=Path().cwd() / "output",
    envvar="OUTPUT_DIR",
    help="the path that you want to save the new model & checkpoints in",
)
@click.option(
    "--split_method",
    default="scaffold",
    type=click.Choice(["random", "scaffold"]),
    envvar="SPLIT_METHOD",
    help="how to split the data if --val_dataset and --test_dataset not given",
)
@click.option(
    "--fit_metric",
    type=str,
    default="loss",
    envvar="FIT_METRIC",
    help="the metric that the early stopping callback will monitor to stop training early",
)
@click.option(
    "--greater_is_better",
    is_flag=True,
    envvar="GREATER_IS_BETTER",
    help="whether a higher or lower number is desired from the val_metric",
)
@click.option(
    "--learning_rate",
    type=float,
    default=1e-6,
    envvar="LEARNING_RATE",
    help="the learning rate to apply to the optimizer",
)
@click.option(
    "--epochs",
    type=int,
    default=100,
    envvar="EPOCHS",
    help="number of epochs to train the model for",
)
@click.option(
    "--patience",
    type=int,
    default=10,
    envvar="PATIENCE",
    help="number of epochs to monitor early stopping for",
)
@click.option(
    "--weight_decay",
    type=float,
    default=1e-2,
    envvar="WEIGHT_DECAY",
    help="the weight decay to apply to the optimizer",
)
@click.option(
    "--batch_size",
    type=int,
    default=8,
    envvar="BATCH_SIZE",
    help="indicate the batch size per device (for both train & eval)",
)
@click.option(
    "--smiles_col",
    type=str,
    default="smiles",
    envvar="SMILES_COL",
    help="column in the dataset that contains the SMILES strings",
)
@click.option(
    "--label_col",
    type=str,
    default="labels",
    envvar="LABEL_COL",
    help="the column of the csv that contains the labels",
)
@click.option(
    "--tf32",
    is_flag=True,
    envvar="TF32",
    help="whether to use tf32 during finetuning",
)
@click.option(
    "--eval_metrics",
    multiple=True,
    type=click.Choice(
        [
            "roc_auc",
            "prc_auc",
            "f1",
            "precision",
            "recall",
            "accuracy",
            "hamming_accuracy",
        ]
    ),
    default=[
        "roc_auc",
        "prc_auc",
        "f1",
        "precision",
        "recall",
        "accuracy",
        "hamming_accuracy",
    ],
    envvar="EVAL_METRICS",
    help="the metrics to report during finetuning",
)
@click.option(
    "--eval_acc_steps",
    type=int,
    default=None,
    envvar="EVAL_ACC_STEPS",
    help="the amount of steps of eval before moving data to CPU",
)
@click.option(
    "--seed", type=int, default=0, envvar="SEED", help="the random seed to use"
)
def main(
    dataset: Path,
    val_dataset: Path,
    test_dataset: Path,
    clean: bool,
    pretrained_model: str,
    output_dir: Path,
    split_method: str,
    fit_metric: str,
    greater_is_better: bool,
    learning_rate: float,
    epochs: int,
    patience: int,
    weight_decay: float,
    batch_size: int,
    smiles_col: str,
    label_col: str,
    tf32: bool,
    eval_metrics: List[str],
    eval_acc_steps: int,
    seed: int,
) -> None:
    """Fine-tune a T5 model for molecular property prediction.

    This function loads a pretrained T5 model, prepares the dataset with
    appropriate tokenization, and trains the model using the HuggingFace
    Seq2SeqTrainer. Training includes early stopping based on the specified
    metric and saves the best model checkpoint.

    After training, the function evaluates on the test set and exports
    predictions and metrics to the output directory.

    Args:
        dataset: Path to the training dataset (CSV file or HuggingFace dataset).
        val_dataset: Optional path to validation dataset. If None, data is split
            according to split_method.
        test_dataset: Optional path to test dataset. If None, data is split
            according to split_method.
        clean: Whether to canonicalize SMILES and remove salts before training.
        pretrained_model: Name or path of the pretrained model to fine-tune.
        output_dir: Directory to save the fine-tuned model, checkpoints, and results.
        split_method: Method for splitting data ('scaffold' or 'random').
        fit_metric: Metric to monitor for early stopping and best model selection.
        greater_is_better: Whether higher values of fit_metric are better.
        learning_rate: Learning rate for the AdamW optimizer.
        epochs: Maximum number of training epochs.
        patience: Number of epochs without improvement before early stopping.
        weight_decay: Weight decay coefficient for regularization.
        batch_size: Batch size per device for training and evaluation.
        smiles_col: Column name containing SMILES strings in the dataset.
        label_col: Column name containing space-separated labels in the dataset.
        tf32: Whether to use TensorFloat-32 precision on supported GPUs.
        eval_metrics: List of metrics to compute during evaluation.
        eval_acc_steps: Steps between moving eval data to CPU (helps with OOM).
        seed: Random seed for reproducibility.

    Output files:
        - {output_dir}/config.json: Model configuration
        - {output_dir}/pytorch_model.bin: Model weights
        - {output_dir}/tokenizer_config.json: Tokenizer configuration
        - {output_dir}/test_predictions.csv: Per-sample predictions on test set
        - {output_dir}/test_metrics.json: Aggregated test set metrics
    """

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        n_devices = torch.cuda.device_count()
        print(f"Found {n_devices} CUDA devices.")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    train_dataset = smiles_t5.dataset.load_dataset(
        dataset_path=dataset,
        val_path=val_dataset,
        test_path=test_dataset,
        smiles_col=smiles_col,
        splitting_method=split_method,
        clean=clean,
    )

    unique_labels = []
    for labels in np.unique(train_dataset["train"][label_col]):
        for label in labels.split(" "):
            if label:
                if label not in unique_labels:
                    unique_labels.append(label)
    tokenizer.add_tokens(unique_labels)

    def tokenize(data):
        results = tokenizer(data[smiles_col])
        results["labels"] = tokenizer(data[label_col])["input_ids"]
        return results

    tokenized = train_dataset.map(
        tokenize,
        batched=True,
        num_proc=1,
        remove_columns=train_dataset["train"].column_names,
    )

    model = T5ForConditionalGeneration.from_pretrained(
        pretrained_model,
    )
    model.config.labels = unique_labels
    model.resize_token_embeddings(
        len(tokenizer),
        pad_to_multiple_of=32 if tf32 else None,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=tokenizer.pad_token_id,
        pad_to_multiple_of=32 if tf32 else None,
    )

    calc_metrics = smiles_t5.metrics.T5Seq2SeqMetricCalculator(
        tokenizer=tokenizer,
        labels=unique_labels,
        metrics=eval_metrics,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        learning_rate=learning_rate,
        lr_scheduler_type="linear",
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=fit_metric,
        greater_is_better=greater_is_better,
        tf32=tf32,
        eval_accumulation_steps=eval_acc_steps,
        seed=seed,
        data_seed=seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],  # type: ignore[arg-type]
        eval_dataset=tokenized["val"],  # type: ignore[arg-type]
        data_collator=data_collator,
        compute_metrics=calc_metrics,  # type: ignore[arg-type]
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=patience,
                early_stopping_threshold=1e-6,
            )
        ],
    )

    trainer.train()
    # save model & tokenizer
    if trainer.model is not None:
        trainer.model.save_pretrained(output_dir)  # type: ignore[union-attr]
    tokenizer.save_pretrained(output_dir)
    # get test data
    predictions = trainer.predict(tokenized["test"])  # type: ignore[arg-type]
    print(f"Test Metric: {predictions.metrics}")
    prediction_exporter = smiles_t5.metrics.PredictionExporter(
        tokenizer=tokenizer,
        labels=model.config.labels,
        path=output_dir,
    )
    prediction_exporter.export_test_predictions(
        predictions.predictions[0],
        cast(np.ndarray, predictions.label_ids),
        tokenized["test"],
    )
    prediction_exporter.export_test_metrics(
        cast(dict, predictions.metrics),
    )


if __name__ == "__main__":
    main()
