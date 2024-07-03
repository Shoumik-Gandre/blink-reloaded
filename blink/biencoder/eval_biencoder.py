# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import logging
import os
from typing import Annotated

import torch
from tqdm import tqdm
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from blink.biencoder.biencoder import BiEncoderRanker, BiEncoderRankerParams
import blink.biencoder.data_process as data
import blink.biencoder.nn_prediction as nnquery
import blink.candidate_ranking.utils as utils
from blink.biencoder.zeshel_utils import WORLDS, load_entity_dict_zeshel
from blink.common.params import set_constant_tokens


def load_entity_dict(logger, params, is_zeshel):
    if is_zeshel:
        return load_entity_dict_zeshel(logger, params, params['entity_dict_path'])

    path = params.get("entity_dict_path", None)
    assert path is not None, "Error! entity_dict_path is empty."

    entity_list = []
    logger.info("Loading entity description from path: " + path)
    with open(path, 'rt') as f:
        for line in f:
            sample = json.loads(line.rstrip())
            title = sample['title']
            text = sample.get("text", "").strip()
            entity_list.append((title, text))
            if params["debug"] and len(entity_list) > 200:
                break

    return entity_list


# zeshel version of get candidate_pool_tensor
def get_candidate_pool_tensor_zeshel(
    entity_dict,
    tokenizer,
    max_seq_length,
    logger,
):
    candidate_pool = {}
    for src in range(len(WORLDS)):
        if entity_dict.get(src, None) is None:
            continue
        logger.info("Get candidate desc to id for pool %s" % WORLDS[src])
        candidate_pool[src] = get_candidate_pool_tensor(
            entity_dict[src],
            tokenizer,
            max_seq_length,
            logger,
        )

    return candidate_pool


def get_candidate_pool_tensor_helper(
    entity_desc_list,
    tokenizer,
    max_seq_length,
    logger,
    is_zeshel,
):
    if is_zeshel:
        return get_candidate_pool_tensor_zeshel(
            entity_desc_list,
            tokenizer,
            max_seq_length,
            logger,
        )
    else:
        return get_candidate_pool_tensor(
            entity_desc_list,
            tokenizer,
            max_seq_length,
            logger,
        )


def get_candidate_pool_tensor(
    entity_desc_list,
    tokenizer,
    max_seq_length,
    logger,
):
    # TODO: add multiple thread process
    logger.info("Convert candidate text to id")
    cand_pool = [] 
    for entity_desc in tqdm(entity_desc_list):
        if type(entity_desc) is tuple:
            title, entity_text = entity_desc
        else:
            title = None
            entity_text = entity_desc

        rep = data.get_candidate_representation(
                entity_text, 
                tokenizer, 
                max_seq_length,
                title,
        )
        cand_pool.append(rep["ids"])

    cand_pool = torch.LongTensor(cand_pool) 
    return cand_pool


def encode_candidate(
    reranker: BiEncoderRanker,
    candidate_pool,
    encode_batch_size,
    silent,
    logger,
    is_zeshel,
):
    if is_zeshel:
        src = 0
        cand_encode_dict = {}
        for src, cand_pool in candidate_pool.items():
            logger.info("Encoding candidate pool %s" % WORLDS[src])
            cand_pool_encode = encode_candidate(
                reranker,
                cand_pool,
                encode_batch_size,
                silent,
                logger,
                is_zeshel=False,
            )
            cand_encode_dict[src] = cand_pool_encode
        return cand_encode_dict
        
    reranker.model.eval()
    device = reranker.device
    sampler = SequentialSampler(candidate_pool)
    data_loader = DataLoader(
        candidate_pool, sampler=sampler, batch_size=encode_batch_size
    )
    if silent:
        iter_ = data_loader
    else:
        iter_ = tqdm(data_loader)

    cand_encode_list = None
    for step, batch in enumerate(iter_):
        cands = batch
        cands = cands.to(device)
        cand_encode = reranker.encode_candidate(cands)
        if cand_encode_list is None:
            cand_encode_list = cand_encode
        else:
            cand_encode_list = torch.cat((cand_encode_list, cand_encode))

    return cand_encode_list


def load_or_generate_candidate_pool(
    tokenizer,
    params,
    logger,
    cand_pool_path,
):
    candidate_pool = None
    is_zeshel = params.get("zeshel", None)
    if cand_pool_path is not None:
        # try to load candidate pool from file
        try:
            logger.info("Loading pre-generated candidate pool from: ")
            logger.info(cand_pool_path)
            candidate_pool = torch.load(cand_pool_path)
        except:
            logger.info("Loading failed. Generating candidate pool")

    if candidate_pool is None:
        # compute candidate pool from entity list
        entity_desc_list = load_entity_dict(logger, params, is_zeshel)
        candidate_pool = get_candidate_pool_tensor_helper(
            entity_desc_list,
            tokenizer,
            params["max_cand_length"],
            logger,
            is_zeshel,
        )

        if cand_pool_path is not None:
            logger.info("Saving candidate pool.")
            torch.save(candidate_pool, cand_pool_path)

    return candidate_pool


import typer
def run(
        output_path: Annotated[str, typer.Argument(help="The output directory where the generated output file (model, etc.) is to be dumped.")],
        eval_batch_size: int = typer.Option(default=8, help="Total batch size for evaluation."),
        mode: str = typer.Option(default="test", help="Train / validation / test"),
        save_topk_result: bool = typer.Option(default=False, help="Whether to save prediction results."),
        encode_batch_size: int = typer.Option(default=8, help="Batch size for encoding."),
        cand_pool_path: str = typer.Option(default=None, help="Path for cached candidate pool (id tokenization of candidates)"),
        entity_dict_path: str = typer.Option(default=None, help="Path for entity dictionary"),
        cand_encode_path: str = typer.Option(default=None, help="Path for cached candidate encoding"),
        roberta: bool = typer.Option(default=False, help="Is the bert model roberta or not"),
        # blink args
        silent: bool = typer.Option(False, help="Whether to print progress bars."),
        debug: bool = typer.Option(False, help="Whether to run in debug mode with only 200 samples."),
        data_parallel: bool = typer.Option(False, help="Whether to distributed the candidate generation process."),
        cuda: bool = typer.Option(False, help="Whether not to use CUDA when available"),
        top_k: int = typer.Option(10),
        seed: int = typer.Option(default=52313, help="random seed for initialization"),
        zeshel: bool = typer.Option(False, help="Whether the dataset is from zeroshot."),
        # model args
        max_seq_length: int = typer.Option(256, help="The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded."),
        max_context_length: int = typer.Option(128, help="The maximum total context input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded."),
        max_cand_length: int = typer.Option(128, help="The maximum total label input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded."),
        path_to_model: str = typer.Option(None, help="The full path to the model to load."),
        bert_model: str = typer.Option("bert-base-uncased", help="Bert pre-trained model selected in the list: bert-base-uncased, bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese."),
        pull_from_layer: int = typer.Option(-1, help="Layers to pull from BERT."),
        lowercase: bool = typer.Option(True, help="Whether to lower case the input text. True for uncased models, False for cased models."),
        context_key: str = typer.Option("context", help="Key for context."),
        out_dim: int = typer.Option(1, help="Output dimension of bi-encoders."),
        add_linear: bool = typer.Option(False, help="Whether to add an additional linear projection on top of BERT."),
        data_path: str = typer.Option("data/zeshel", help="The path to the train data."),
    ) -> None:
    set_constant_tokens({'roberta': roberta})
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logger = utils.get_logger(output_path)

    biencoder_ranker_params: BiEncoderRankerParams = {
        "bert_model": bert_model,
        "out_dim": out_dim,
        "pull_from_layer": pull_from_layer,
        "add_linear": add_linear,
        "no_cuda": not cuda,
        "lowercase": lowercase,
        "path_to_model": path_to_model,
        "data_parallel": data_parallel,
    }

    # Init model 
    reranker = BiEncoderRanker(biencoder_ranker_params)
    tokenizer = reranker.tokenizer
    
    # candidate encoding is not pre-computed. 
    # load/generate candidate pool to compute candidate encoding.
    candidate_pool = load_or_generate_candidate_pool(
        tokenizer,
        {
            'zeshel': zeshel,
            'max_cand_length': max_cand_length,
            'entity_dict_path': entity_dict_path,
            'debug': debug,
            'mode': mode
        },
        logger,
        cand_pool_path,
    )       

    candidate_encoding = None
    if cand_encode_path is not None:
        # try to load candidate encoding from path
        # if success, avoid computing candidate encoding
        try:
            logger.info("Loading pre-generated candidate encode path.")
            candidate_encoding = torch.load(cand_encode_path)
        except:
            logger.info("Loading failed. Generating candidate encoding.")

    if candidate_encoding is None:
        candidate_encoding = encode_candidate(
            reranker,
            candidate_pool,
            encode_batch_size,
            silent=silent,
            logger=logger,
            is_zeshel=zeshel 
        )

        if cand_encode_path is not None:
            # Save candidate encoding to avoid re-compute
            logger.info("Saving candidate encoding to file " + cand_encode_path)
            torch.save(candidate_encoding, cand_encode_path)
    
    def map_function(sample):
        return data.process_mention_data(
            sample,
            tokenizer,
            max_context_length,
            max_cand_length,
            context_key=context_key,
        )
    
    def create_tensor_dataset(tensor_dataset):
        context_ids = torch.stack([x for x in tensor_dataset["context_ids"]])
        label_ids = torch.stack([x for x in tensor_dataset["label_ids"]])
        label_idx = torch.stack([x for x in tensor_dataset["label_idx"]])
        if "src" in tensor_dataset.column_names:
            src = torch.stack([x["src"] for x in tensor_dataset])
            return TensorDataset(context_ids, label_ids, src, label_idx)
        else:
            return TensorDataset(context_ids, label_ids, label_idx)
        
    test_samples: Dataset = load_dataset('shomez/zeshel-blink', split=mode)

    logger.info("Read %d test samples." % len(test_samples))
    test_tensor_data = (test_samples.map(map_function, batched=False, num_proc=4, desc="Representation: "))
    test_tensor_data.set_format(type='torch', columns=['context_ids','label_ids','label_idx','src',])
    test_tensor_data = create_tensor_dataset(test_tensor_data)

    test_sampler = SequentialSampler(test_tensor_data)
    test_dataloader = DataLoader(test_tensor_data, sampler=test_sampler, batch_size=eval_batch_size)
    
    new_data = nnquery.get_topk_predictions(
        reranker,
        test_dataloader,
        candidate_pool,
        candidate_encoding,
        silent,
        logger,
        top_k,
        zeshel,
        save_topk_result,
    )

    if save_topk_result: 
        save_data_dir = os.path.join(
            output_path,
            "top%d_candidates" % top_k,
        )
        if not os.path.exists(save_data_dir):
            os.makedirs(save_data_dir)
        save_data_path = os.path.join(save_data_dir, "%s.t7" % mode)
        torch.save(new_data, save_data_path)


if __name__ == "__main__":
    app = typer.Typer(add_completion=False, pretty_exceptions_show_locals = False)
    app.command()(run)
    app()