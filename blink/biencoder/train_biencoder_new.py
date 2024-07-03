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
    WEIGHTS_NAME, 
    CONFIG_NAME, 
    PreTrainedTokenizer, 
    Trainer, 
    TrainingArguments
)
from datasets import load_dataset, Dataset

from blink.biencoder.biencoder import (
    BiEncoderRanker, 
    BiEncoderModule, 
    BiEncoderRankerParams
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


def save_model(model: BiEncoderModule, tokenizer: PreTrainedTokenizer, output_dir: str):
    """Saves the model and the tokenizer used in the output directory."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model_to_save = model.module if hasattr(model, "module") else model
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)
    tokenizer.save_pretrained(output_dir)
    model_to_save.cand_encoder.bert_model.save_pretrained(str(Path(output_dir) / "entity-encoder"))
    model_to_save.context_encoder.bert_model.save_pretrained(str(Path(output_dir) / "mention-encoder"))


def run(
    silent: bool = typer.Option(False, help="Whether to print progress bars."),
    debug: bool = typer.Option(False, help="Whether to run in debug mode with only 200 samples."),
    data_parallel: bool = typer.Option(False, help="Whether to distribute the candidate generation process."),
    no_cuda: bool = typer.Option(False, help="Whether not to use CUDA when available"),
    top_k: int = typer.Option(10),
    seed: int = typer.Option(52313, help="Random seed for initialization"),
    zeshel: bool = typer.Option(False, help="Whether the dataset is from zeroshot."),
    max_seq_length: int = typer.Option(256, help="The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded."),
    max_context_length: int = typer.Option(128, help="The maximum total context input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded."),
    max_cand_length: int = typer.Option(128, help="The maximum total label input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded."),
    path_to_model: str = typer.Option(None, help="The full path to the model to load."),
    bert_model: str = typer.Option("bert-base-uncased", help="Bert pre-trained model selected in the list: bert-base-uncased, bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese."),
    pull_from_layer: int = typer.Option(-1, help="Layers to pull from BERT"),
    lowercase: bool = typer.Option(True, help="Whether to lower case the input text. True for uncased models, False for cased models."),
    out_dim: int = typer.Option(1, help="Output dimension of bi-encoders."),
    add_linear: bool = typer.Option(False, help="Whether to add an additional linear projection on top of BERT."),
    data_path: str = typer.Option("data/zeshel", help="The path to the train data."),
    output_path: str = typer.Option(..., help="The output directory where the generated output file (model, etc.) is to be dumped."),
    run_evaluate: bool = typer.Option(False, help="Whether to run evaluation."),
    output_eval_file: str = typer.Option(None, help="The txt file where the evaluation results will be written."),
    train_batch_size: int = typer.Option(8, help="Total batch size for training."),
    max_grad_norm: float = typer.Option(1.0),
    learning_rate: float = typer.Option(3e-5, help="The initial learning rate for Adam."),
    num_train_epochs: int = typer.Option(1, help="Number of training epochs."),
    print_interval: int = typer.Option(10, help="Interval of loss printing"),
    eval_interval: int = typer.Option(100, help="Interval for evaluation during training"),
    save_interval: int = typer.Option(1, help="Interval for model saving"),
    warmup_proportion: float = typer.Option(0.1, help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training."),
    gradient_accumulation_steps: int = typer.Option(1, help="Number of updates steps to accumulate before performing a backward/update pass."),
    type_optimization: str = typer.Option("all_encoder_layers", help="Which type of layers to optimize in BERT"),
    shuffle: bool = typer.Option(False, help="Whether to shuffle train data"),
    roberta: bool = typer.Option(False, help="Is the bert model roberta or not"),
    eval_batch_size: int = typer.Option(8, help="Total batch size for evaluation."),
    mode: str = typer.Option("valid", help="Train / validation / test"),
    save_topk_result: bool = typer.Option(False, help="Whether to save prediction results."),
    encode_batch_size: int = typer.Option(8, help="Batch size for encoding."),
    cand_pool_path: str = typer.Option(None, help="Path for cached candidate pool (id tokenization of candidates)"),
    entity_dict_path: str = typer.Option(None, help="Path for entity dictionary"),
    cand_encode_path: str = typer.Option(None, help="Path for cached candidate encoding"),
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

    # Init model
    reranker = BiEncoderRanker(
        BiEncoderRankerParams(
            bert_model=bert_model,
            out_dim=out_dim,
            pull_from_layer=pull_from_layer,
            add_linear=add_linear,
            no_cuda=no_cuda,
            lowercase=lowercase,
            path_to_model=path_to_model,
            data_parallel=data_parallel,
        )
    )
    tokenizer = reranker.tokenizer
    tokenizer.add_special_tokens({'additional_special_tokens': ['[unused0]', '[unused1]', '[unused2]']})
    # Entity split token should be a special token in tokenizer
    assert '[unused0]' in tokenizer.all_special_tokens
    assert '[unused1]' in tokenizer.all_special_tokens
    assert '[unused2]' in tokenizer.all_special_tokens

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
            'mention_start_token': '[unused0]',
            'mention_end_token': '[unused1]',
            'entity_sep_token': '[unused2]',
        }, 
        batched=False, 
        num_proc=4, 
        desc="Representation: "
    )
    train_samples.set_format(type='torch', columns=pt_columns)
    logger.info("Read %d train samples." % len(train_samples))

    valid_samples: Dataset = load_dataset('shomez/zeshel-blink', split='validation')
    valid_samples = valid_samples.map(
        tokenize_all, 
        fn_kwargs={
            'tokenizer': tokenizer,
            'max_context_length': max_context_length,
            'max_cand_length': max_cand_length,
            'mention_start_token': '[unused0]',
            'mention_end_token': '[unused1]',
            'entity_sep_token': '[unused2]',
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
        do_train=True,
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=num_train_epochs,
        max_grad_norm=max_grad_norm,
        learning_rate=learning_rate,
        warmup_ratio=warmup_proportion
    )

    trainer = Trainer(
        model=reranker,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_samples,
        eval_dataset=valid_samples,
        tokenizer=reranker.tokenizer,
    )
    trainer.train()


if __name__ == "__main__":
    app = typer.Typer(add_completion=False, pretty_exceptions_show_locals = False)
    app.command()(run)
    app()