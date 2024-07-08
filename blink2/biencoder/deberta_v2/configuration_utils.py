from transformers import (
    PretrainedConfig, 
    DebertaV2Config, 
    AutoConfig,
)


class DebertaV2BiencoderRankerConfig(PretrainedConfig):
    model_type = "deberta-v2"
    is_composition = True
    
    def __init__(
            self, 
            mention_encoder: DebertaV2Config,
            entity_encoder: DebertaV2Config,
            embed_dim: int=384, 
            **kwargs
        ):
        super().__init__(**kwargs)
        
        self.embed_dim = embed_dim
        
        if isinstance(mention_encoder, PretrainedConfig):
            self.mention_encoder = mention_encoder
        else:
            self.mention_encoder = AutoConfig.for_model(**mention_encoder)

        if isinstance(entity_encoder, PretrainedConfig):
            self.entity_encoder = entity_encoder
        else:
            self.entity_encoder = AutoConfig.for_model(**entity_encoder)

        self.mention_encoder.embed_dim = embed_dim
        self.entity_encoder.embed_dim = embed_dim 