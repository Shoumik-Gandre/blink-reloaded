# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
from pathlib import Path
from typing import Dict, TypedDict
import torch
import random
import numpy as np
import logging

import typer
from transformers import (
    Trainer, 
    TrainingArguments,
    DebertaV2Config,
    AutoTokenizer
)
from datasets import load_dataset, Dataset

from blink.biencoder.biencoder2 import (
    DebertaV2BiencoderRankerConfig,
    DebertaV2BiencoderRanker
)
import blink.candidate_ranking.utils as utils
from blink.biencoder.data_process2 import tokenize_all


logger = None
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'


def collate_fn(sample):
    mention_input_ids = torch.stack([x["mention_input_ids"] for x in sample])
    mention_attention_mask = torch.stack([x["mention_attention_mask"] for x in sample])
    mention_token_type_ids = torch.stack([x["mention_token_type_ids"] for x in sample])

    entity_input_ids = torch.stack([x["entity_input_ids"] for x in sample])
    entity_attention_mask = torch.stack([x["entity_attention_mask"] for x in sample])
    entity_token_type_ids = torch.stack([x["entity_token_type_ids"] for x in sample])

    result = {
        'mention_input_ids': mention_input_ids,
        'mention_attention_mask': mention_attention_mask,
        'mention_token_type_ids': mention_token_type_ids,
        'entity_input_ids': entity_input_ids,
        'entity_attention_mask': entity_attention_mask,
        'entity_token_type_ids': entity_token_type_ids,
    }
    
    return result


def run(
    output_path: str = typer.Option(..., help="The output directory where the generated output file (model, etc.) is to be dumped."),
    data_path: str = typer.Option("data/zeshel", help="The path to the train data."),
    embed_dim: int = typer.Option(300, help="Output dimension of bi-encoders."),
    max_context_length: int = typer.Option(128, help="The maximum total context input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded."),
    max_cand_length: int = typer.Option(128, help="The maximum total label input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded."),
    num_train_epochs: int = typer.Option(1, help="Number of training epochs."),
    train_batch_size: int = typer.Option(8, help="Total batch size for training."),
    learning_rate: float = typer.Option(3e-5, help="The initial learning rate for Adam."),
    max_grad_norm: float = typer.Option(1.0),
    warmup_proportion: float = typer.Option(0.1, help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training."),
    gradient_accumulation_steps: int = typer.Option(1, help="Number of updates steps to accumulate before performing a backward/update pass."),
    debug: bool = typer.Option(False, help="Whether to run in debug mode with only 200 samples."),
    seed: int = typer.Option(52313, help="Random seed for initialization"),
):
    # Fix the random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)

    model_output_path = output_path
    Path(model_output_path).mkdir(exist_ok=True, parents=True)
    logger = utils.get_logger(output_path)

    MODEL_NAME = "microsoft/deberta-v3-xsmall"

    # Init model
    mention_encoder_config = DebertaV2Config.from_pretrained(MODEL_NAME)
    entity_encoder_config = DebertaV2Config.from_pretrained(MODEL_NAME)

    reranker_config = DebertaV2BiencoderRankerConfig(
        mention_encoder=mention_encoder_config,
        entity_encoder=entity_encoder_config,
        embed_dim=embed_dim
    )
    reranker = DebertaV2BiencoderRanker(reranker_config)
    reranker.mention_encoder.from_pretrained(MODEL_NAME, config=mention_encoder_config)
    reranker.entity_encoder.from_pretrained(MODEL_NAME, config=entity_encoder_config)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens({'additional_special_tokens': ['[B-MEN]', '[E-MEN]', '[ENT]']})

    pt_columns = [
        "mention_input_ids", 
        "mention_attention_mask", 
        "mention_token_type_ids", 
        "entity_input_ids", 
        "entity_attention_mask", 
        "entity_token_type_ids", 
        'label_idx',
        'src'
    ]
    # Load train data
    train_samples: Dataset = load_dataset('shomez/zeshel-blink', split='train')
    if debug: 
        train_samples = train_samples.shuffle(seed=seed).select(range(200))
    train_samples = train_samples.map(
        tokenize_all, 
        fn_kwargs={
            'tokenizer': tokenizer,
            'max_context_length': max_context_length,
            'max_cand_length': max_cand_length,
            'mention_start_token': '[B-MEN]',
            'mention_end_token': '[E-MEN]',
            'entity_sep_token': '[ENT]',
        }, 
        batched=False, 
        num_proc=4, 
        desc="Representation: "
    )
    train_samples.set_format(type='torch', columns=pt_columns)
    logger.info("Read %d train samples." % len(train_samples))

    valid_samples: Dataset = load_dataset('shomez/zeshel-blink', split='validation')
    if debug: 
        valid_samples = valid_samples.shuffle(seed=seed).select(range(10))
    valid_samples = valid_samples.map(
        tokenize_all, 
        fn_kwargs={
            'tokenizer': tokenizer,
            'max_context_length': max_context_length,
            'max_cand_length': max_cand_length,
            'mention_start_token': '[B-MEN]',
            'mention_end_token': '[E-MEN]',
            'entity_sep_token': '[ENT]',
        }, 
        batched=False, 
        num_proc=4, 
        desc="Representation: "
    )
    valid_samples.set_format(type='torch', columns=pt_columns)
    logger.info("Read %d valid samples." % len(valid_samples))

    # evaluate before training
    training_args = TrainingArguments(
        output_dir=output_path,
        overwrite_output_dir=True,
        do_train=True,
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        # eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=num_train_epochs,
        max_grad_norm=max_grad_norm,
        learning_rate=learning_rate,
        warmup_ratio=warmup_proportion,
        fp16=True,
        # save_strategy='steps',
        # save_steps=10, 
        # max_steps=10
    )

    trainer = Trainer(
        model=reranker,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_samples,
        # eval_dataset=valid_samples,
        tokenizer=tokenizer,
    )
    trainer.train()


if __name__ == "__main__":
    app = typer.Typer(add_completion=False, pretty_exceptions_show_locals = False)
    app.command()(run)
    app()