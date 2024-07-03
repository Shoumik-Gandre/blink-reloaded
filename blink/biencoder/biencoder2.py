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
from transformers import AutoModel, PreTrainedModel, PretrainedConfig


class BiEncoderRankerConfig(PretrainedConfig):
    model_type = "biencoder_ranker"
    
    def __init__(self, mention_encoder: str, entity_encoder: str, **kwargs):
        super().__init__(**kwargs)
        pass
    

class BiEncoderRanker(PreTrainedModel):
    """
    This is a wrapper class for training mention_encoder and entity_encoder
    """
    def __init__(self, config: BiEncoderRankerConfig, mention_encoder: nn.Module, entity_encoder: nn.Module):
        super().__init__(config)
        self.mention_encoder = mention_encoder
        self.entity_encoder = entity_encoder

    def encode_mentions(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor) -> torch.Tensor:
        return self.mention_encoder(input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]

    def encode_entities(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor, token_type_ids: torch.LongTensor) -> torch.Tensor:
        return self.entity_encoder(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state[:, 0, :]

    # Score candidates given context input and label input
    # If cand_encs is provided (pre-computed), cand_ves is ignored
    def similarity(
            self,
            mention_embeddings: torch.Tensor,
            entity_embeddings: torch.LongTensor,
            random_negs=True,
    ):  
        # Train time. We compare with all elements of the batch
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
            entity_input_ids: torch.LongTensor,
            entity_attention_mask: torch.LongTensor,
            entity_token_type_ids: torch.LongTensor,
            labels: torch.LongTensor, #! Hard negative input labels
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        mention_embeddings = self.encode_mentions(mention_input_ids, mention_attention_mask)
        entity_embeddings = self.encode_entities(entity_input_ids, entity_attention_mask, entity_token_type_ids)
        scores = self.similarity(mention_embeddings, entity_embeddings, labels is None)
        if labels is None:
            target = torch.arange(scores.size(0), dtype=torch.long, device=scores.device)
            loss = F.cross_entropy(scores, target, reduction="mean")
        else:
            loss_fct = nn.BCEWithLogitsLoss(reduction="mean")
            loss = loss_fct(scores, labels)
        return loss, scores
