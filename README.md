# SmilesT5

## Environment

```
# create the conda environment
conda create -n smilest5 python=3.10

# activate it
conda activate smilest5

# install the required packages
pip install -r requirements.txt
```

## Finetuning
Example finetuning command:

``` 
python finetune.py \
    --dataset dataset.csv \
    --output_dir path/to/output/dir \
    --split_method scaffold \
    --label_col labels \
    --eval_metrics "roc_auc "f1_score" \
    --fit_metric "roc_auc" \
    --greater_is_better \
    --tf32
```

| argument | default | type | description |
|----------|---------|------|-------------|
| --dataset |  | string | The path to the dataset that will be used to train the model. This can be either a .csv file or a huggingface dataset/datasetdict directory |
| --val_dataset |  | string |  optional path to the validation datatset, if not given, the dataset will be split according to --split_method |
| --test_dataset |  | string |  optional path to the test datatset, if not given, the dataset will be split according to --split_method |
| --clean | false | string | whether or not to clean the SMILES strings before training. This will remove salts and canonicalise them. |
| --pretrained_model | hhtx/smiles_t5 | string |  the name (or path if stored locally) of the model to load |
| --output_dir |  | string |  the path that you want to save the new model & final test metrics in |
| --split_method | scaffold | string |  how to split the data if --val_dataset and --test_dataset not given, this can be scaffold or random splitting. **Allowed values: scaffold, random** |
| --fit_metrics | loss | str | the metric that the early stopping callback will monitor to stop training early |
| --greater_is_better | false | bool | whether a higher or lower number is desired from the val_metric |
| --learning_rate | 1e-6 | float |  the learning rate to apply to the optimizer |
| --epochs | 100 | int |  number of epochs to train the model for |
| --patience | 10 | int |  number of epochs to monitor early stopping for |
| --weight_decay | 1e-2 | float |  the weight decay to apply to the optimizer |
| --batch_size | 8 | int |  indicate the batch size per device (for both train & eval) |
| --smiles_col | smiles | string |  column in the dataset that contains the SMILES strings |
| --label_col | labels | string |  the column of the dataset that contains the labels |
| --dataset_col | | string |  the column to use to split the dataset into train/val/test |
| --tf32 | false | bool |  whether to use tf32 during training [[read more here](https://huggingface.co/docs/transformers/v4.41.3/en/perf_train_gpu_one#tf32)] |
| --eval_metrics | "roc_auc" "prc_auc" "f1" "precision" "recall" "accuracy" "hamming_accuracy" | strings | which metrics to calculate against the test dataset |
| --eval_acc_steps | None | int |  the amount of steps during evaluation before moving data to the CPU, can slow down performance but helps to limit CUDA OOM errors for large datasets |
| --seed | 0 | int |  the random seed to use |


## Getting Predictions

``` 
python predict.py \
    --model_dir path/to/finetuned/model \
    --dataset path/to/dataset.csv \
    --output_file path/to/outputs.csv \
    --smiles_col "smiles" \
    --batch_size 8
```

| argument | default | type | description |
|----------|---------|------|-------------|
| --dataset |  | string | The path to the dataset that will be used to train the model. This can be either a .csv file or a huggingface dataset/datasetdict directory |
| --model_dir | | str | the path to the finetuned model & tokenizer |
| --smiles_col | smiles | str | column in the dataset that contains the SMILES strings |
| --batch_size | 1 | int | the batch size used for inference |
| --clean | false | bool | whether to clean the smiles strings before generating predictions |
| --output_file |  | str | the CSV file to save the data in, if not given, it will be in the current directory |

## Extracting Embeddings

``` 
python extract_embeddings.py \
    --dataset path/to/dataset.csv \
    --smiles_col "smiles"
```

| argument | default | type | description |
|----------|---------|------|-------------|
| --pretrained_model | hhtx/smiles_t5 | string |  the name (or path if stored locally) of the model to load |
| --dataset |  | string | The path to the dataset that will be used to train the model. This can be either a .csv file or a huggingface dataset/datasetdict directory |
| --smiles_col | smiles | str | column in the dataset that contains the SMILES strings |
| --batch_size | 8 | int |  indicate the batch size per device (for both train & eval) |
| --clean | false | bool | whether to clean the smiles strings before generating predictions |

