from typing import NamedTuple, Dict
from transformers import AutoModel
from transformers.pipelines import PIPELINE_REGISTRY, Pipeline
from transformers.pipelines.base import build_pipeline_init_args, GenericTensor
from transformers.utils import add_end_docstrings
from transformers.modeling_outputs import BaseModelOutputWithPooling


class EntityInput(NamedTuple):
    text: str
    title: str


@add_end_docstrings(
    build_pipeline_init_args(has_tokenizer=True, supports_binary_output=False),
    r"""
        tokenize_kwargs (`dict`, *optional*):
                Additional dictionary of keyword arguments passed along to the tokenizer.
        return_tensors (`bool`, *optional*):
            If `True`, returns a tensor according to the specified framework, otherwise returns a list.""",
)
class EntityEncoderPipeline(Pipeline):
    """
    Feature extraction pipeline uses no model head. This pipeline extracts the hidden states from the base
    transformer, which can be used as features in downstream tasks.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> extractor = pipeline(model=shomez/blink-biencoder-mention-encoder"", tokenizer="google-bert/bert-large-uncased", task="feature-extraction")
    >>> result = extractor("This is a simple test.", return_tensors=True)
    >>> result.shape  # This is a tensor of shape [1, hidden_dimension] representing the input string.
    torch.Size([1, 1024])
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This feature extraction pipeline can currently be loaded from [`pipeline`] using the task identifier:
    `"feature-extraction"`.

    All models may be used for this pipeline. See a list of all models, including community-contributed models on
    [huggingface.co/models](https://huggingface.co/models).
    """

    def _sanitize_parameters(self, truncation=None, tokenize_kwargs=None, return_tensors=None, **kwargs):
        if tokenize_kwargs is None:
            tokenize_kwargs = {}

        if truncation is not None:
            if "truncation" in tokenize_kwargs:
                raise ValueError(
                    "truncation parameter defined twice (given as keyword argument as well as in tokenize_kwargs)"
                )
            tokenize_kwargs["truncation"] = truncation

        preprocess_params = tokenize_kwargs

        postprocess_params = {}
        if return_tensors is not None:
            postprocess_params["return_tensors"] = return_tensors

        return preprocess_params, {}, postprocess_params

    def preprocess(self, inputs: EntityInput, **tokenize_kwargs) -> Dict[str, GenericTensor]:
        input_string = f"{inputs.title} [unused2] {inputs.text}"
        model_inputs = self.tokenizer(
            input_string, 
            return_tensors=self.framework, 
            return_token_type_ids=True,
            **tokenize_kwargs
        )
        return model_inputs

    def _forward(self, model_inputs):
        model_outputs = self.model(**model_inputs)
        return model_outputs

    def postprocess(self, model_outputs: BaseModelOutputWithPooling, return_tensors=False):
        if return_tensors:
            return model_outputs.last_hidden_state[:, 0, :].squeeze(0)
        if self.framework == "pt":
            return model_outputs.last_hidden_state[:, 0, :].squeeze(0).tolist()
        elif self.framework == "tf":
            return model_outputs.last_hidden_state[:, 0, :].numpy().tolist()

    def __call__(self, *args, **kwargs):
        """
        Extract the features of the input(s).

        Args:
            args (`EntityInput` or `List[EntityInput]`): One or several texts (or one list of texts) to get the features of.

        Return:
            A nested list of `float`: The features computed by the model.
        """
        return super().__call__(*args, **kwargs)


PIPELINE_REGISTRY.register_pipeline(
    "entity-encoder",
    pipeline_class=EntityEncoderPipeline,
    pt_model=AutoModel,
    default={
        'model': {
            "pt": ('shomez/blink-biencoder-description-encoder', "")
        }
    },
    type="text",  # current support type: text, audio, image, multimodal
)