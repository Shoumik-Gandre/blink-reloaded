from .entity_encoder_pipeline import EntityEncoderPipeline, EntityInput
from .mention_encoder_pipeline import MentionEncoderPipeline
from transformers.pipelines import PIPELINE_REGISTRY
from transformers import AutoModel


def register_mention_encoder_pipeline():
    PIPELINE_REGISTRY.register_pipeline(
        "mention-encoder",
        pipeline_class=MentionEncoderPipeline,
        pt_model=AutoModel,
        default={
            'model': {
                "pt": ('shomez/blink-biencoder-mention-encoder', "")
            }
        },
        type="text",  # current support type: text, audio, image, multimodal
    )


def register_entity_encoder_pipeline():
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


register_mention_encoder_pipeline()
register_entity_encoder_pipeline()