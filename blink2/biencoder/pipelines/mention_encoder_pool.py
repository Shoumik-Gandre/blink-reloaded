from typing import Dict, List

from transformers.pipelines import Pipeline
from transformers.pipelines.base import build_pipeline_init_args, GenericTensor
from transformers.utils import add_end_docstrings
from transformers.modeling_outputs import BaseModelOutputWithPooling

from ..preprocess import MentionInput, tokenize_mention


@add_end_docstrings(
    build_pipeline_init_args(has_tokenizer=True, supports_binary_output=False),
    r"""
        tokenize_kwargs (`dict`, *optional*):
                Additional dictionary of keyword arguments passed along to the tokenizer.
        return_tensors (`bool`, *optional*):
            If `True`, returns a tensor according to the specified framework, otherwise returns a list.""",
)
class MentionEncoderPoolPipeline(Pipeline):

    def _sanitize_parameters(self, truncation=None, tokenize_kwargs=None, return_tensors=None, max_length=None, **kwargs):
        if tokenize_kwargs is None:
            tokenize_kwargs = {}

        if truncation is not None:
            if "truncation" in tokenize_kwargs:
                raise ValueError(
                    "truncation parameter defined twice (given as keyword argument as well as in tokenize_kwargs)"
                )
            tokenize_kwargs["truncation"] = truncation
            tokenize_kwargs["max_length"] = max_length

        preprocess_params = tokenize_kwargs

        postprocess_params = {}
        if return_tensors is not None:
            postprocess_params["return_tensors"] = return_tensors

        return preprocess_params, {}, postprocess_params

    def preprocess(self, inputs: MentionInput | List[MentionInput], **tokenize_kwargs) -> Dict[str, GenericTensor]:
        model_inputs = tokenize_mention(
            inputs,
            tokenizer=self.tokenizer,
            return_tensors=self.framework,
            prepend_batch_axis=True,
            **tokenize_kwargs
        )
        return model_inputs

    def _forward(self, model_inputs):
        model_outputs = self.model(**model_inputs).pooler_output
        return model_outputs

    def postprocess(self, model_outputs: BaseModelOutputWithPooling, return_tensors=False):
        if return_tensors:
            return model_outputs.squeeze(0)
        if self.framework == "pt":
            return model_outputs.squeeze(0).tolist()
        elif self.framework == "tf":
            return model_outputs.numpy().tolist()

    def __call__(self, *args, **kwargs):
        """
        Extract the features of the input(s).

        Args:
            args (`str` or `List[str]`): One or several texts (or one list of texts) to get the features of.

        Return:
            A nested list of `float`: The features computed by the model.
        """
        return super().__call__(*args, **kwargs)