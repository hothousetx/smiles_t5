import transformers
import torch
from typing import Dict, OrderedDict

import transformers.modeling_outputs

class T5Seq2SeqPipeline(transformers.Text2TextGenerationPipeline):
    def __init__(self, *args, **kwargs):
        super(T5Seq2SeqPipeline, self).__init__(*args, **kwargs)
        self.labels = self.model.config.labels
        self.label_ids = self.tokenizer.convert_tokens_to_ids(self.labels)

    def _forward(self, model_inputs: Dict[str, torch.Tensor], **generate_kwargs) -> transformers.generation.utils.GenerateEncoderDecoderOutput:
        input_length = model_inputs["input_ids"].shape[1]

        generate_kwargs["min_length"] = 0
        generate_kwargs["max_length"] = len(self.labels) + 1  # +1 to compensate for the pad token
        generate_kwargs["return_dict_in_generate"] = True
        generate_kwargs["output_scores"] = True
        self.check_inputs(input_length, generate_kwargs["min_length"], generate_kwargs["max_length"])

        outputs = self.model.generate(**model_inputs, **generate_kwargs)
        return outputs

    def postprocess(self, model_outputs: OrderedDict[str, torch.Tensor]) -> Dict[str, float]:
        stacked_scores = torch.vstack(model_outputs.scores).softmax(dim=1)
        max_scores = stacked_scores.max(dim=0).values
        label_scores = [i.item() for i in max_scores[self.label_ids]]
        return dict(zip(self.labels, label_scores))