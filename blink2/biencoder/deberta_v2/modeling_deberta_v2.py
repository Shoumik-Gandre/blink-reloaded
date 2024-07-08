# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
from typing import Any, Dict, Optional, Tuple, TypedDict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2PreTrainedModel,
    DebertaV2Model,
    ContextPooler,
    StableDropout
)

from .configuration_utils import DebertaV2BiencoderRankerConfig


class DebertaV2ForTextEncoding(DebertaV2PreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)

        embed_dim = config.embed_dim        
        self.embed_dim = embed_dim

        self.deberta = DebertaV2Model(config)
        self.pooler = ContextPooler(config)
        output_dim = self.pooler.output_dim

        self.classifier = nn.Linear(output_dim, embed_dim)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = StableDropout(drop_out)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self): return self.deberta.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings): self.deberta.set_input_embeddings(new_embeddings)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = self.pooler(encoder_layer)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return BaseModelOutputWithPooling(
            last_hidden_state=outputs,
            pooler_output=logits,
            hidden_states=outputs.hidden_states, 
            attentions=outputs.attentions
        )       


class DebertaV2BiencoderRanker(PreTrainedModel):
    """
    This is a wrapper class for training mention_encoder and entity_encoder.
    """
    config_class=DebertaV2BiencoderRankerConfig
    
    def __init__(self, config: DebertaV2BiencoderRankerConfig):
        super().__init__(config)
        self.mention_encoder = DebertaV2ForTextEncoding(config.mention_encoder)
        self.entity_encoder = DebertaV2ForTextEncoding(config.entity_encoder)
        self.config = config

    def encode_mentions(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor, token_type_ids: torch.LongTensor) -> torch.Tensor:
        outputs: BaseModelOutputWithPooling = self.mention_encoder(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return outputs.pooler_output

    def encode_entities(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor, token_type_ids: torch.LongTensor) -> torch.Tensor:
        outputs: BaseModelOutputWithPooling = self.entity_encoder(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return outputs.pooler_output

    def similarity(self, mention_embeddings: torch.Tensor, entity_embeddings: torch.LongTensor, random_negs=True) -> torch.Tensor:
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
