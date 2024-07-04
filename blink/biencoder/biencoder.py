# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import Optional, TypedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, BertModel

from blink.common.ranker_base import BertEncoder
from blink.common.params import BERT_START_TOKEN, BERT_END_TOKEN


class BiEncoderModuleParams(TypedDict):
    bert_model: str
    out_dim: int
    pull_from_layer: int
    add_linear: bool


def load_biencoder(params: BiEncoderModuleParams):
    # Init model
    biencoder = BiEncoderRanker(params)
    return biencoder


class BiEncoderModule(torch.nn.Module):
    def __init__(self, params: BiEncoderModuleParams):
        super(BiEncoderModule, self).__init__()
        ctxt_bert = BertModel.from_pretrained(params["bert_model"])
        cand_bert = BertModel.from_pretrained(params['bert_model'])
        self.context_encoder = BertEncoder(ctxt_bert, params["out_dim"], layer_pulled=params["pull_from_layer"], add_linear=params["add_linear"])
        self.cand_encoder = BertEncoder(cand_bert, params["out_dim"], layer_pulled=params["pull_from_layer"], add_linear=params["add_linear"])
        self.config = ctxt_bert.config

    def forward(
            self,
            token_idx_ctxt,
            segment_idx_ctxt,
            mask_ctxt,
            token_idx_cands,
            segment_idx_cands,
            mask_cands,
    ):
        embedding_ctxt = None
        if token_idx_ctxt is not None:
            embedding_ctxt = self.context_encoder(token_idx_ctxt, segment_idx_ctxt, mask_ctxt)
        embedding_cands = None
        if token_idx_cands is not None:
            embedding_cands = self.cand_encoder(token_idx_cands, segment_idx_cands, mask_cands)
        return embedding_ctxt, embedding_cands


class BiEncoderRankerParams(TypedDict):
    bert_model: str
    out_dim: int
    pull_from_layer: int
    add_linear: bool
    no_cuda: bool
    lowercase: bool
    data_parallel: bool
    path_to_model: Optional[str]
    

class BiEncoderRanker(torch.nn.Module):
    def __init__(self, params: BiEncoderRankerParams):
        super(BiEncoderRanker, self).__init__()
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() and not params["no_cuda"] else "cpu")
        # init tokenizer
        self.NULL_IDX = 0
        self.START_TOKEN = BERT_START_TOKEN
        self.END_TOKEN = BERT_END_TOKEN
        self.tokenizer = AutoTokenizer.from_pretrained(params["bert_model"], do_lower_case=params["lowercase"])
        # init model
        self.build_model()
        model_path = params.get("path_to_model", None)
        if model_path is not None:
            self.load_model(model_path)

        self.model = self.model.to(self.device)
        self.data_parallel = params.get("data_parallel")
        if self.data_parallel:
            self.model = torch.nn.DataParallel(self.model)

    def load_model(self, fname):
        state_dict = torch.load(fname, map_location=self.device)
        self.model.load_state_dict(state_dict)

    def build_model(self):
        self.model = BiEncoderModule(self.params)
    
    def score_candidate(
            self,
            mention_input_ids: torch.LongTensor, 
            mention_attention_mask: torch.LongTensor, 
            mention_token_type_ids: torch.LongTensor,
            entity_input_ids: torch.LongTensor, 
            entity_attention_mask: torch.LongTensor, 
            entity_token_type_ids: torch.LongTensor, 
            random_negs=True,
            cand_encs=None,  # pre-computed candidate encoding.
    ) -> torch.Tensor:
        # Encode contexts first
        embedding_ctxt, _ = self.model(mention_input_ids, mention_token_type_ids, mention_attention_mask, None, None, None)

        # Candidate encoding is given, do not need to re-compute
        # Directly return the score of context encoding and candidate encoding
        if cand_encs is not None:
            return embedding_ctxt.mm(cand_encs.t())

        _, embedding_cands = self.model(None, None, None, entity_input_ids, entity_token_type_ids, entity_attention_mask)
        
        if random_negs:
            # train on random negatives
            embedding_ctxt: torch.Tensor
            embedding_cands: torch.Tensor
            return embedding_ctxt.mm(embedding_cands.t())
        else:
            # train on hard negatives
            embedding_ctxt = embedding_ctxt.unsqueeze(1)  # batchsize x 1 x embed_size
            embedding_cands = embedding_cands.unsqueeze(2)  # batchsize x embed_size x 1
            scores = torch.bmm(embedding_ctxt, embedding_cands)  # batchsize x 1 x 1
            scores = torch.squeeze(scores)
            return scores

    def forward(
            self, 
            mention_input_ids: torch.LongTensor, 
            mention_attention_mask: torch.LongTensor, 
            mention_token_type_ids: torch.LongTensor,
            entity_input_ids: torch.LongTensor, 
            entity_attention_mask: torch.LongTensor, 
            entity_token_type_ids: torch.LongTensor, 
            label_input=None
        ):
        flag = label_input is None
        scores = self.score_candidate(
            mention_input_ids, 
            mention_attention_mask, 
            mention_token_type_ids,
            entity_input_ids, 
            entity_attention_mask, 
            entity_token_type_ids, 
            flag
        )
        bs = scores.size(0)
        if label_input is None:
            target = torch.arange(bs, dtype=torch.long, device=scores.device)
            loss = F.cross_entropy(scores, target, reduction="mean")
        else:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(scores, label_input)
        return loss, scores
    
    def encode_context(self, cands):
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(cands, self.NULL_IDX)
        embedding_context, _ = self.model(token_idx_cands, segment_idx_cands, mask_cands, None, None, None)
        return embedding_context.cpu().detach()

    def encode_candidate(self, cands):
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(cands, self.NULL_IDX)
        _, embedding_cands = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )
        # TODO: why do we need cpu here?
        return embedding_cands.cpu().detach()


def to_bert_input(token_idx, null_idx):
    """ token_idx is a 2D tensor int.
        return token_idx, segment_idx and mask
    """
    segment_idx = token_idx * 0
    mask = token_idx != null_idx
    # nullify elements in case self.NULL_IDX was not 0
    token_idx = token_idx * mask.long()
    return token_idx, segment_idx, mask
