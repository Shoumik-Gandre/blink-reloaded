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
import time
import numpy as np
import logging

import typer
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, TensorDataset
from transformers import WEIGHTS_NAME, CONFIG_NAME, get_linear_schedule_with_warmup, PreTrainedTokenizer
from datasets import load_dataset, Dataset

from blink.biencoder.biencoder import BiEncoderRanker, load_biencoder, BiEncoderModule, BiEncoderRankerParams
import blink.candidate_ranking.utils as utils
from blink.biencoder.data_process2 import tokenize_all
from blink.common.optimizer import get_bert_optimizer


logger = None
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

# The evaluate function during training uses in-batch negatives:
# for a batch of size B, the labels from the batch are used as label candidates
# B is controlled by the parameter eval_batch_size
def evaluate(reranker: BiEncoderRanker, eval_dataloader: DataLoader, logger: logging.Logger, silent: bool=False) -> Dict[str, float]:
    reranker.model.eval()

    eval_accuracy = 0.0
    nb_eval_examples = 0
    nb_eval_steps = 0

    for step, batch in enumerate(tqdm(eval_dataloader, desc="Evaluation", disable=silent)):
        batch = tuple(t.to(reranker.device) for t in batch)
        (
            mention_input_ids, 
            mention_attention_mask, 
            mention_token_type_ids, 
            entity_input_ids, 
            entity_attention_mask, 
            entity_token_type_ids, 
            *_
        ) = batch

        with torch.no_grad():
            eval_loss, logits = reranker(
                mention_input_ids, 
                mention_attention_mask, 
                mention_token_type_ids, 
                entity_input_ids, 
                entity_attention_mask, 
                entity_token_type_ids, 
            )

        logits = logits.detach().cpu().numpy()
        # Using in-batch negatives, the label ids are diagonal
        label_ids = np.arange(logits.shape[0])
        tmp_eval_accuracy, _ = utils.accuracy(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += mention_input_ids.size(0)
        nb_eval_steps += 1

    normalized_eval_accuracy = eval_accuracy / nb_eval_examples
    logger.info("Eval accuracy: %.5f" % normalized_eval_accuracy)
    return {"normalized_accuracy": normalized_eval_accuracy}


def get_optimizer(model, type_optimization, learning_rate, fp16):
    return get_bert_optimizer(
        [model],
        type_optimization,
        learning_rate,
        fp16=fp16,
    )


def get_scheduler(params, optimizer, len_train_data, logger):
    batch_size = params["train_batch_size"]
    grad_acc = params["gradient_accumulation_steps"]
    epochs = params["num_train_epochs"]

    num_train_steps = int(len_train_data / batch_size / grad_acc) * epochs
    num_warmup_steps = int(num_train_steps * params["warmup_proportion"])

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
    logger.info(" Num optimization steps = %d" % num_train_steps)
    logger.info(" Num warmup steps = %d", num_warmup_steps)
    return scheduler


# If you need a DataLoader, you can convert the dataset to a TensorDataset
def create_tensor_dataset(tensor_dataset: Dataset):
    context_ids = torch.stack([x for x in tensor_dataset["context_ids"]])
    label_ids = torch.stack([x for x in tensor_dataset["label_ids"]])
    label_idx = torch.stack([x for x in tensor_dataset["label_idx"]])
    if "src" in tensor_dataset.column_names:
        src = torch.stack([x["src"] for x in tensor_dataset])
        return TensorDataset(context_ids, label_ids, src, label_idx)
    else:
        return TensorDataset(context_ids, label_ids, label_idx)
    

def create_tensor_dataset2(tensor_dataset: Dataset):
    mention_input_ids = torch.stack([x for x in tensor_dataset["mention_input_ids"]])
    mention_attention_mask = torch.stack([x for x in tensor_dataset["mention_attention_mask"]])
    mention_token_type_ids = torch.stack([x for x in tensor_dataset["mention_token_type_ids"]])

    entity_input_ids = torch.stack([x for x in tensor_dataset["entity_input_ids"]])
    entity_attention_mask = torch.stack([x for x in tensor_dataset["entity_attention_mask"]])
    entity_token_type_ids = torch.stack([x for x in tensor_dataset["entity_token_type_ids"]])

    label_idx = torch.stack([x for x in tensor_dataset["label_idx"]])
    if "src" in tensor_dataset.column_names:
        src = torch.stack([x["src"] for x in tensor_dataset])
        return TensorDataset(mention_input_ids, mention_attention_mask, mention_token_type_ids, entity_input_ids, entity_attention_mask, entity_token_type_ids, src, label_idx)
    else:
        return TensorDataset(mention_input_ids, mention_attention_mask, mention_token_type_ids, entity_input_ids, entity_attention_mask, entity_token_type_ids, label_idx)


class CollateFnInput(TypedDict):
    context_ids: list
    label_ids: list
    label_idx: list


def collate_fn(sample: CollateFnInput):
    mention_input_ids = torch.stack([x for x in sample["mention_input_ids"]])
    mention_attention_mask = torch.stack([x for x in sample["mention_attention_mask"]])
    mention_token_type_ids = torch.stack([x for x in sample["mention_token_type_ids"]])

    entity_input_ids = torch.stack([x for x in sample["entity_input_ids"]])
    entity_attention_mask = torch.stack([x for x in sample["entity_attention_mask"]])
    entity_token_type_ids = torch.stack([x for x in sample["entity_token_type_ids"]])
    label_idx = torch.stack([x for x in sample["label_idx"]])

    result = {
        'mention_input_ids': mention_input_ids,
        'mention_attention_mask': mention_attention_mask,
        'mention_token_type_ids': mention_token_type_ids,
        'entity_input_ids': entity_input_ids,
        'entity_attention_mask': entity_attention_mask,
        'entity_token_type_ids': entity_token_type_ids,
        'label_idx': label_idx,
    }
    if "src" in sample.keys():
        src = torch.stack([x["src"] for x in sample])
        result['src'] = src
    
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
    context_key: str = typer.Option("context"),
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
    params = {
        "silent": silent,
        "debug": debug,
        "data_parallel": data_parallel,
        "no_cuda": no_cuda,
        "top_k": top_k,
        "seed": seed,
        "zeshel": zeshel,
        "max_seq_length": max_seq_length,
        "max_context_length": max_context_length,
        "max_cand_length": max_cand_length,
        "path_to_model": path_to_model,
        "bert_model": bert_model,
        "pull_from_layer": pull_from_layer,
        "lowercase": lowercase,
        "context_key": context_key,
        "out_dim": out_dim,
        "add_linear": add_linear,
        "data_path": data_path,
        "output_path": output_path,
        "evaluate": run_evaluate,
        "output_eval_file": output_eval_file,
        "train_batch_size": train_batch_size,
        "max_grad_norm": max_grad_norm,
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "print_interval": print_interval,
        "eval_interval": eval_interval,
        "save_interval": save_interval,
        "warmup_proportion": warmup_proportion,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "type_optimization": type_optimization,
        "shuffle": shuffle,
        "roberta": roberta,
        "eval_batch_size": eval_batch_size,
        "mode": mode,
        "save_topk_result": save_topk_result,
        "encode_batch_size": encode_batch_size,
        "cand_pool_path": cand_pool_path,
        "entity_dict_path": entity_dict_path,
        "cand_encode_path": cand_encode_path,
        "fp16": False
    }

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

    model = reranker.model

    device = reranker.device
    n_gpu = torch.cuda.device_count()

    if gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                gradient_accumulation_steps
            )
        )

    # An effective batch size of `x`, when we are accumulating the gradient accross `y` batches will be achieved by having a batch size of `z = x / y`
    # args.gradient_accumulation_steps = args.gradient_accumulation_steps // n_gpu
    train_batch_size = train_batch_size // gradient_accumulation_steps
    grad_acc_steps = gradient_accumulation_steps

    # Fix the random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

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
    train_samples.set_format(type='torch', columns=pt_columns)

    train_tensor_data = create_tensor_dataset2(train_samples)
    logger.info("Read %d train samples." % len(train_samples))

    train_dataloader = DataLoader(train_tensor_data, shuffle=shuffle, batch_size=train_batch_size)

    # Load eval data
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
    valid_tensor_data = create_tensor_dataset2(valid_samples)
    logger.info("Read %d valid samples." % len(valid_samples))

    valid_dataloader = DataLoader(valid_tensor_data, shuffle=False, batch_size=eval_batch_size)

    # evaluate before training
    # results = evaluate(reranker, valid_dataloader, logger=logger, silent=silent)

    time_start = time.time()

    utils.write_to_file(str(Path(model_output_path) / "training_params.txt"), str(params))

    logger.info("Starting training")
    logger.info("device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, False))

    optimizer = get_optimizer(model, params['type_optimization'], params['learning_rate'], params['fp16'])
    scheduler = get_scheduler(params, optimizer, len(train_tensor_data), logger)

    model.train()

    best_epoch_idx = -1
    best_score = -1

    for epoch_idx in trange(int(num_train_epochs), desc="Epoch"):
        tr_loss = 0
        results = None

        for step, batch in enumerate(tqdm(train_dataloader, desc="Batch", disable=silent)):
            batch = tuple(t.to(device) for t in batch)
            (
                mention_input_ids, 
                mention_attention_mask, 
                mention_token_type_ids, 
                entity_input_ids, 
                entity_attention_mask, 
                entity_token_type_ids, 
                *_
            ) = batch
            loss, _ = reranker(
                mention_input_ids, 
                mention_attention_mask, 
                mention_token_type_ids, 
                entity_input_ids, 
                entity_attention_mask, 
                entity_token_type_ids, 
            )

            if grad_acc_steps > 1:
                loss = loss / grad_acc_steps

            tr_loss += loss.item()

            if (step + 1) % (print_interval * grad_acc_steps) == 0:
                logger.info(
                    "Step {} - epoch {} average loss: {}\n".format(
                        step,
                        epoch_idx,
                        tr_loss / (print_interval * grad_acc_steps),
                    )
                )
                tr_loss = 0

            loss.backward()

            if (step + 1) % grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % (eval_interval * grad_acc_steps) == 0:
                logger.info("Evaluation on the development dataset")
                evaluate(reranker, valid_dataloader, logger=logger, silent=silent)
                model.train()
                logger.info("\n")

        logger.info("***** Saving fine - tuned model *****")
        epoch_output_folder_path = Path(model_output_path) / f"epoch_{epoch_idx}"
        model.eval()
        save_model(model, tokenizer, str(epoch_output_folder_path))

        results = evaluate(reranker, valid_dataloader, logger=logger, silent=silent)

        ls = [best_score, results["normalized_accuracy"]]
        li = [best_epoch_idx, epoch_idx]

        best_score = ls[np.argmax(ls)]
        best_epoch_idx = li[np.argmax(ls)]
        logger.info("\n")

    execution_time = (time.time() - time_start) / 60
    utils.write_to_file(
        os.path.join(model_output_path, "training_time.txt"),
        "The training took {} minutes\n".format(execution_time),
    )
    logger.info("The training took {} minutes\n".format(execution_time))

    # save the best model in the parent_dir
    logger.info("Best performance in epoch: {}".format(best_epoch_idx))
    params["path_to_model"] = str(Path(model_output_path) /  "epoch_{}".format(best_epoch_idx) / WEIGHTS_NAME)
    reranker = load_biencoder(params)
    model.eval()
    save_model(reranker.model, tokenizer, model_output_path)

    if run_evaluate:
        params["path_to_model"] = model_output_path
        evaluate(reranker, valid_dataloader, logger=logger, silent=silent)


if __name__ == "__main__":
    app = typer.Typer(add_completion=False, pretty_exceptions_show_locals = False)
    app.command()(run)
    app()