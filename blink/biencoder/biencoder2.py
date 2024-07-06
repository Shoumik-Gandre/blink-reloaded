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
from transformers import (
    PreTrainedModel, 
    PretrainedConfig, 
    DebertaV2ForSequenceClassification, 
    DebertaV2Config, 
    AutoConfig,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import (
    MSELoss,
    CrossEntropyLoss,
    BCEWithLogitsLoss
)
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2PreTrainedModel,
    DebertaV2Model,
    ContextPooler,
    StableDropout
)


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
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
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

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    # regression task
                    loss_fn = nn.MSELoss()
                    logits = logits.view(-1).to(labels.dtype)
                    loss = loss_fn(logits, labels.view(-1))
                elif labels.dim() == 1 or labels.size(-1) == 1:
                    label_index = (labels >= 0).nonzero()
                    labels = labels.long()
                    if label_index.size(0) > 0:
                        labeled_logits = torch.gather(
                            logits, 0, label_index.expand(label_index.size(0), logits.size(1))
                        )
                        labels = torch.gather(labels, 0, label_index.view(-1))
                        loss_fct = CrossEntropyLoss()
                        loss = loss_fct(labeled_logits.view(-1, self.num_labels).float(), labels.view(-1))
                    else:
                        loss = torch.tensor(0).to(logits)
                else:
                    log_softmax = nn.LogSoftmax(-1)
                    loss = -((log_softmax(logits) * labels).sum(-1)).mean()
            elif self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
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
        print(f"Inside config {self.embed_dim = } {self.mention_encoder.embed_dim}")
        

class DebertaV2BiencoderRanker(PreTrainedModel):
    """
    This is a wrapper class for training mention_encoder and entity_encoder.
    The Deberta Encoders are implemented with DebertaV2ForSequenceClassifiers. 
    This is a workaround because it has the exact architecture required.
    Ideally, we define a seperate derived class for it: DebertaV2Encoder
    """
    config_class=DebertaV2BiencoderRankerConfig
    
    def __init__(self, config: DebertaV2BiencoderRankerConfig):
        super().__init__(config)
        self.mention_encoder = DebertaV2ForTextEncoding(config.mention_encoder)
        self.entity_encoder = DebertaV2ForTextEncoding(config.entity_encoder)
        self.config = config

    def encode_mentions(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor, token_type_ids: torch.LongTensor) -> torch.Tensor:
        outputs: SequenceClassifierOutput = self.mention_encoder(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return outputs.logits

    def encode_entities(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor, token_type_ids: torch.LongTensor) -> torch.Tensor:
        outputs: SequenceClassifierOutput = self.entity_encoder(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return outputs.logits

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
