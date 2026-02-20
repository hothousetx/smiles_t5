This repository contains scripts for fine-tuning and inference of the SmilesT5 model, a transformer-based approach for learning molecular properties from SMILES representations. SmilesT5 is introduced in our paper:

**[SmilesT5: Domain-specific pretraining for molecular language models](#)**

> Molecular property prediction is an increasingly critical task within drug discovery and development. Typically, neural networks can learn molecular properties using graph-based, language-based or feature-based methods. Recent advances in natural language processing have highlighted the capabilities of neural networks to learn complex human language using masked language modelling. These approaches to training large transformer-based deep learning models have also been used to learn the language of molecules, as represented by simplified molecular-input line-entry system (SMILES) strings. Here, we present novel domain-specific text-to-text pretraining tasks that yield improved performance in six classification-based molecular property prediction benchmarks, relative to both traditional likelihood-based training and previously proposed fine-tuning tasks. Through ablation studies, we show that data and computational efficiency can be improved by using these domain-specific pretraining tasks. Finally, the pretrained embeddings from the model can be used as fixed inputs into a downstream machine learning classifier and yield comparable performance to finetuning but with much lower computational overhead.

## Environment

```bash
# create the conda environment
conda create -n smilest5 python=3.10

# activate it
conda activate smilest5

# install the required packages
pip install -r requirements.txt
```

## Package Structure

The `smiles_t5` package contains the following modules:

- **`dataset`**: Data loading, preprocessing, and splitting utilities for molecular datasets. Supports scaffold and random splitting methods.
- **`pipeline`**: Inference pipeline for generating predictions from SMILES strings using fine-tuned models.
- **`metrics`**: Evaluation metrics (ROC-AUC, F1, etc.) and prediction export utilities for seq2seq models.

## Fine-tuning

Fine-tune a pretrained SmilesT5 model on your molecular property prediction task:

```bash
python finetune.py \
    --dataset dataset.csv \
    --output_dir path/to/output/dir \
    --split_method scaffold \
    --label_col labels \
    --eval_metrics roc_auc f1 \
    --fit_metric roc_auc \
    --greater_is_better \
    --tf32
```

### Fine-tuning Arguments

| Argument              | Default                | Type  | Description                                                                                    |
| --------------------- | ---------------------- | ----- | ---------------------------------------------------------------------------------------------- |
| `--dataset`           | _required_             | Path  | Path to the training dataset (CSV file or HuggingFace dataset directory)                       |
| `--val_dataset`       | None                   | Path  | Optional path to validation dataset. If not given, data is split according to `--split_method` |
| `--test_dataset`      | None                   | Path  | Optional path to test dataset. If not given, data is split according to `--split_method`       |
| `--clean`             | False                  | Flag  | Whether to canonicalize SMILES and remove salts before training                                |
| `--pretrained_model`  | `hothousetx/smiles_t5` | str   | Name or path of the pretrained model to fine-tune                                              |
| `--output_dir`        | `./output`             | Path  | Directory to save the fine-tuned model, checkpoints, and results                               |
| `--split_method`      | `scaffold`             | str   | Method for splitting data: `scaffold` or `random`                                              |
| `--fit_metric`        | `loss`                 | str   | Metric to monitor for early stopping and best model selection                                  |
| `--greater_is_better` | False                  | Flag  | Whether higher values of `--fit_metric` are better                                             |
| `--learning_rate`     | `1e-6`                 | float | Learning rate for the AdamW optimizer                                                          |
| `--epochs`            | `100`                  | int   | Maximum number of training epochs                                                              |
| `--patience`          | `10`                   | int   | Number of epochs without improvement before early stopping                                     |
| `--weight_decay`      | `1e-2`                 | float | Weight decay coefficient for regularization                                                    |
| `--batch_size`        | `8`                    | int   | Batch size per device for training and evaluation                                              |
| `--smiles_col`        | `smiles`               | str   | Column name containing SMILES strings in the dataset                                           |
| `--label_col`         | `labels`               | str   | Column name containing space-separated labels in the dataset                                   |
| `--tf32`              | False                  | Flag  | Whether to use TensorFloat-32 precision on supported GPUs                                      |
| `--eval_metrics`      | All metrics            | str   | Metrics to compute during evaluation (can specify multiple)                                    |
| `--eval_acc_steps`    | None                   | int   | Steps between moving eval data to CPU (helps prevent OOM errors)                               |
| `--seed`              | `0`                    | int   | Random seed for reproducibility                                                                |

### Available Evaluation Metrics

- `roc_auc`: Area under the ROC curve
- `prc_auc`: Area under the Precision-Recall curve
- `f1`: F1 score at threshold 0.5
- `precision`: Precision at threshold 0.5
- `recall`: Recall at threshold 0.5
- `accuracy`: Exact match accuracy across all labels
- `hamming_accuracy`: Hamming accuracy (per-label accuracy averaged)

### Output Files

After fine-tuning, the following files are saved to `--output_dir`:

- `config.json`: Model configuration
- `pytorch_model.bin`: Model weights
- `tokenizer_config.json`: Tokenizer configuration
- `test_predictions.csv`: Per-sample predictions on the test set
- `test_metrics.json`: Aggregated test set metrics

## Getting Predictions

Generate predictions from a fine-tuned model with optional Monte Carlo dropout for uncertainty estimation:

```bash
python predict.py \
    --model_dir path/to/finetuned/model \
    --dataset path/to/dataset.csv \
    --output_file path/to/outputs.csv \
    --smiles_col smiles \
    --batch_size 8 \
    --mc_dropout_samples 10
```

### Prediction Arguments

| Argument               | Default    | Type | Description                                                                          |
| ---------------------- | ---------- | ---- | ------------------------------------------------------------------------------------ |
| `--dataset`            | _required_ | Path | Path to the dataset containing SMILES strings                                        |
| `--model_dir`          | _required_ | Path | Path to the fine-tuned model and tokenizer                                           |
| `--smiles_col`         | `smiles`   | str  | Column name containing SMILES strings in the dataset                                 |
| `--batch_size`         | `1`        | int  | Batch size for inference                                                             |
| `--clean`              | False      | Flag | Whether to canonicalize SMILES before generating predictions                         |
| `--output_file`        | Auto       | Path | Output CSV file path. If not given, saves to `{dataset_stem}_predictions.csv`        |
| `--mc_dropout_samples` | `10`       | int  | Number of Monte Carlo dropout passes for uncertainty estimation. Set to 1 to disable |

### Monte Carlo Dropout

When `--mc_dropout_samples > 1`, the model performs multiple forward passes with dropout enabled to estimate prediction uncertainty. This outputs:

- `{label}_0`, `{label}_1`, ..., `{label}_n`: Individual predictions from each MC dropout pass
- `{label}_mean`: Mean prediction across all MC samples
- `{label}_std`: Standard deviation (uncertainty) across MC samples
- `confidence`: Overall confidence score (1 - mean std across all labels)

When `--mc_dropout_samples = 1`, only the raw predictions for each label are output.

## Extracting Embeddings

Extract molecular embeddings from SMILES strings for use in downstream ML tasks:

```bash
python extract_embeddings.py \
    --dataset path/to/dataset.csv \
    --smiles_col smiles \
    --batch_size 32
```

### Embedding Arguments

| Argument             | Default                | Type | Description                                                 |
| -------------------- | ---------------------- | ---- | ----------------------------------------------------------- |
| `--dataset`          | _required_             | Path | Path to the dataset containing SMILES strings               |
| `--pretrained_model` | `hothousetx/smiles_t5` | str  | Name or path of the pretrained model to use                 |
| `--smiles_col`       | `smiles`               | str  | Column name containing SMILES strings in the dataset        |
| `--batch_size`       | `1`                    | int  | Batch size for embedding extraction                         |
| `--clean`            | False                  | Flag | Whether to canonicalize SMILES before extracting embeddings |

### Output Format

Embeddings are saved as a PyTorch `.pt` file in the same directory as the input dataset, with the same stem name. The file contains a dictionary mapping SMILES strings to their mean-pooled embedding tensors:

```python
import torch

embeddings = torch.load("molecules.pt")
# {'CCO': tensor([...]), 'c1ccccc1': tensor([...]), ...}
```

## Python API

You can also use the package directly in Python:

```python
import smiles_t5

# Load a fine-tuned model
pipeline = smiles_t5.pipeline.T5Seq2SeqPipeline("path/to/model")

# Generate predictions
smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]
predictions = pipeline(smiles_list)

for smiles, pred in zip(smiles_list, predictions):
    print(f"{smiles}: {pred}")
# CCO: {'toxic': 0.12, 'non_toxic': 0.88}
# ...
```

## Citation

If you use SmilesT5 in your research, please cite our paper:

```bibtex
@article{spence2025smilest5,
  title={SmilesT5: Domain-specific pretraining for molecular language models},
  author={Spence, Philip and Paige, Brooks and Osbourn, Anne},
  journal={arXiv preprint},
  year={2025}
}
```

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
