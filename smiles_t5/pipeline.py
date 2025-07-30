from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional, List, Sequence, Union

class T5Seq2SeqPipeline:
    def __init__(
        self,
        pretrained_path: Optional[str] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        model: Optional[T5ForConditionalGeneration] = None,
        device_map: str = "auto",
    ):
        if pretrained_path is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
            self.model = T5ForConditionalGeneration.from_pretrained(pretrained_path)
        else:
            if tokenizer is None or model is None:
                raise ValueError("Either pretrained_path or both tokenizer and model must be provided.")
            self.tokenizer = tokenizer
            self.model = model

        if device_map in ["cuda", "cpu", "mps"]:
            self.device = device_map
        elif device_map == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            raise ValueError("device_map must be 'auto', 'cpu', 'mps', or 'cuda'.")
        self.model.eval()
        self.model.to(self.device) # type: ignore

        self.labels = self.model.config.labels
        self.label_ids = self.tokenizer.convert_tokens_to_ids(self.labels)

    def __call__(self, inputs: Union[Sequence[str], Dataset], **generate_kwargs) -> List[Dict[str, float]]:
        if isinstance(inputs, str):
            inputs = [inputs]
        elif isinstance(inputs, Dataset):
            inputs = [i for i in inputs]
        elif isinstance(inputs, list):
            for i in inputs:
                if not isinstance(i, str):
                    raise ValueError("If inputs is a list, all elements must be strings.")
        else:
            raise ValueError("Inputs must be a string, list of strings, or a Dataset.")

        model_inputs = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        generate_kwargs["min_length"] = 0
        generate_kwargs["max_length"] = len(self.labels) + 1  # +1 to compensate for the pad token
        generate_kwargs["return_dict_in_generate"] = True
        generate_kwargs["output_scores"] = True

        model_outputs = self.model.generate(**model_inputs, **generate_kwargs)
        output_scores = torch.stack(model_outputs.scores, dim=0).permute(1, 0, 2) # type: ignore

        softmax_scores = output_scores.softmax(dim=-1)
        max_scores = softmax_scores.max(dim=1).values
        label_scores = max_scores[:, self.label_ids]
        scores: List[Dict[str, float]] = []
        for i in range(len(inputs)):
            scores_dict: Dict[str, float] = dict(zip(self.labels, label_scores[i].tolist()))
            scores.append(scores_dict)
        return scores
