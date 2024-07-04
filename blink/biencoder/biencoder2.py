# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
from typing import Optional, Tuple, TypedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig, DebertaForSequenceClassification, DebertaConfig
from transformers.modeling_outputs import SequenceClassifierOutput


class DebertaBiencoderRankerConfig(PretrainedConfig):
    model_type = "deberta_biencoder_ranker"
    
    def __init__(self, mention_encoder_config: DebertaConfig, entity_encoder_config: DebertaConfig, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.mention_encoder_config = mention_encoder_config
        self.entity_encoder_config = entity_encoder_config
        self.mention_encoder_config.num_labels = self.embed_dim
        self.entity_encoder_config.num_labels = self.embed_dim

class DebertaBiencoderRanker(PreTrainedModel):
    """
    This is a wrapper class for training mention_encoder and entity_encoder
    """
    def __init__(self, config: DebertaBiencoderRankerConfig):
        super().__init__(config)
        self.mention_encoder = DebertaForSequenceClassification(config.mention_encoder_config)
        self.entity_encoder = DebertaForSequenceClassification(config.entity_encoder_config)

    def encode_mentions(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor, token_type_ids: torch.LongTensor) -> torch.Tensor:
        outputs: SequenceClassifierOutput = self.mention_encoder(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return outputs.logits

    def encode_entities(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor, token_type_ids: torch.LongTensor) -> torch.Tensor:
        outputs: SequenceClassifierOutput = self.entity_encoder(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return outputs.logits

    def similarity(
            self,
            mention_embeddings: torch.Tensor,
            entity_embeddings: torch.LongTensor,
            random_negs=True,
    ):  
        if random_negs:
            # train on random negatives
            return mention_embeddings.mm(entity_embeddings.t()) # mention_embedding_dim x entity_embedding_dim
        else:
            # train on hard negatives
            mention_embeddings = mention_embeddings.unsqueeze(1)  # batchsize x 1 x embed_size
            entity_embeddings = entity_embeddings.unsqueeze(2)  # batchsize x embed_size x 1
            scores = torch.bmm(mention_embeddings, entity_embeddings)  # batchsize x 1 x 1
            scores = torch.squeeze(scores)
            return scores

    # label_input -- negatives provided
    # If label_input is None, train on in-batch negatives
    def forward(
            self, 
            mention_input_ids: torch.LongTensor,
            mention_attention_mask: torch.LongTensor,
            mention_token_type_ids: torch.LongTensor,
            entity_input_ids: torch.LongTensor,
            entity_attention_mask: torch.LongTensor,
            entity_token_type_ids: torch.LongTensor,
            labels: Optional[torch.LongTensor] = None, #! Hard negative input labels
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        mention_embeddings = self.encode_mentions(mention_input_ids, mention_attention_mask, mention_token_type_ids)
        entity_embeddings = self.encode_entities(entity_input_ids, entity_attention_mask, entity_token_type_ids)
        scores = self.similarity(mention_embeddings, entity_embeddings, labels is None)
        if labels is None:
            target = torch.arange(scores.size(0), dtype=torch.long, device=scores.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(scores, target)
        else:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(scores, labels)
        return loss, scores
