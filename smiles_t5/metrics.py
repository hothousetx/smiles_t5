from typing import List, Dict
import torch
from pathlib import Path
from transformers import T5TokenizerFast
from datasets import Dataset
from torcheval.metrics import functional as metrics
import pandas as pd
import json
import numpy as np

class T5Seq2SeqMetricCalculator:
    def __init__(self, tokenizer: T5TokenizerFast, labels: List[str], metrics: List[str] = ["roc_auc"]):
        self.tokenizer = tokenizer
        self.labels = labels
        self.label_token_ids = self.tokenizer.convert_tokens_to_ids(self.labels)
        self.n_labels = len(self.labels)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.metrics = metrics

    def __call__(self, data):
        # make results dict
        results = {}
        logits, labels = data
        # logits are (generated_tokens, last_hidden_state)
        # take just the generated tokens
        # shape = (n_batch, sequence_length, vocab_size)
        logits = torch.tensor(logits[0])
        # labels are (n_batch, n_tokens + </s>)
        labels = torch.tensor(labels)
        if -100 in labels:
            # this shouldn't be necessary but DataCollatorForSeq2Seq doesn't use label_pad_token_id properly
            # replace -100 with the pad token id
            labels[labels == -100] = self.tokenizer.pad_token_id
        # get the number of labels OR outputted sequence length, which ever is lowest
        min_labels = min(self.n_labels, logits.shape[1] - 1) # shape-1 as EOS shouldn't count
        # softmax over all tokens
        probas = logits.softmax(dim=-1)
        # get the logits for the full sequence (excl. EOS token) and only the label tokens
        probas = probas[:, :min_labels, self.label_token_ids]
        # get the max for each label token
        probas_max = probas.max(dim=1).values
        # LABELS make empty binary matrix of all tokens in the vocab
        binary_labels = torch.zeros(labels.shape[0], len(self.tokenizer))
        # scatter 1 values into where the labels are given
        # https://stackoverflow.com/questions/68274722/why-does-torch-scatter-requires-a-smaller-shape-for-indices-than-values
        # binary_labels.scatter_(1, labels, torch.ones_like(binary_labels))
        rows = torch.arange(0, binary_labels.shape[0])[:, None]
        binary_labels[rows.repeat(1, labels.shape[1]), labels] = 1
        # make a matrix of just the possible predicted labels (n_batch, n_labels)
        binary_labels = binary_labels[:, self.label_token_ids]
        for metric in self.metrics:
            if metric == 'roc_auc':
                # have to transpose probas and labels as each task is taken independently, then mean across them
                results['roc_auc'] = torch.stack([metrics.binary_auroc(probas_max[:, i], binary_labels[:, i]) for i in range(self.n_labels)]).mean().item()
            elif metric == 'prc_auc':
                results['prc_auc'] = torch.stack([metrics.binary_auprc(probas_max[:, i], binary_labels[:, i]) for i in range(self.n_labels)]).mean().item()
            elif metric == 'f1':
                results['f1'] = torch.stack([metrics.binary_f1_score(probas_max[:, i], binary_labels[:, i], threshold=0.5) for i in range(self.n_labels)]).mean().item()
            elif metric == 'precision':
                results['precision'] = torch.stack([metrics.binary_precision(probas_max[:, i].round().int(), binary_labels[:, i].int(), threshold=0.5) for i in range(self.n_labels)]).mean().item()
            elif metric == 'recall':
                results['recall'] = torch.stack([metrics.binary_recall(probas_max[:, i].round().int(), binary_labels[:, i].int(), threshold=0.5) for i in range(self.n_labels)]).mean().item()
            elif metric == 'accuracy':
                results['accuracy'] = metrics.multilabel_accuracy(probas_max, binary_labels, threshold=0.5, criteria="exact_match").item()
            elif metric == 'hamming_accuracy':
                results['hamming_accuracy'] = metrics.multilabel_accuracy(probas_max, binary_labels, threshold=0.5, criteria="hamming").item()
        return results
    
class PredictionExporter:
    def __init__(self, tokenizer: T5TokenizerFast, labels: List[str], path: Path):
        self.tokenizer = tokenizer
        self.labels = labels
        self.label_token_ids = self.tokenizer.convert_tokens_to_ids(self.labels)
        self.n_labels = len(self.labels)
        self.eos_token_id = self.tokenizer.eos_token_id
        self.path = path

    def export_test_predictions(self, logits: np.ndarray, labels: np.ndarray, test_dataset: Dataset):
        logits = torch.tensor(logits)
        # PROBAS softmax the logits to get probability of all tokens
        probas = logits.softmax(dim=-1)
        # get the max probability of each token regardless of position in the prediction
        probas_max = probas.max(dim=1).values
        # get the max probability of just the defined label tokens (n_batch, n_labels)
        probas_max = probas_max[:, self.label_token_ids]
        # LABELS
        labels = torch.tensor(labels)
        if -100 in labels:
            # this shouldn't be necessary but DataCollatorForSeq2Seq doesn't use label_pad_token_id properly
            # replace -100 with the pad token id
            labels[labels == -100] = self.tokenizer.pad_token_id
        # make empty binary matrix of all tokens in the vocab
        binary_labels = torch.zeros(labels.shape[0], len(self.tokenizer))
        # scatter 1 values into where the labels are given
        binary_labels.scatter_(1, labels, torch.ones_like(binary_labels))
        # make a matrix of just the possible predicted labels (n_batch, n_labels)
        binary_labels = binary_labels[:, self.label_token_ids]

        decoded = [
            d.replace(" ", "") 
            for d in self.tokenizer.batch_decode(test_dataset['input_ids'], skip_special_tokens=True)
        ]
        data = dict(smiles=decoded)
        for idx, label in enumerate(self.labels):
            data[f"{label}_probas"] = probas_max[:, idx].numpy()
            data[f"{label}_label"] = binary_labels[:, idx].numpy()
        
        df = pd.DataFrame.from_dict(data)
        df.to_csv(Path(self.path, 'test_predictions.csv'), index=False)
    
    def export_test_metrics(self, test_metrics: Dict[str, float]):
        json.dump(test_metrics, Path(self.path, 'test_metrics.json').open(mode='w'), indent=True)