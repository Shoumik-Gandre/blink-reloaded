from logging import Logger
from typing import List, NamedTuple, Optional, Tuple, Dict, Any, TypedDict

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer

from blink.biencoder import BiEncoderRanker
from blink.crossencoder import CrossEncoderRanker
from blink.biencoder.data_process import process_mention_data
from blink.crossencoder.data_process import filter_crossencoder_tensor_input, prepare_crossencoder_candidates, prepare_crossencoder_mentions
from blink.indexer.faiss_indexer import DenseIndexer


class Mention(TypedDict):
    sent_idx: int
    start_pos: int
    end_pos: int
    text: str
    

class MentionAnnotation(TypedDict):
    label: str
    label_id: int
    context_left: str
    context_right: str
    mention: str
    start_pos: int
    end_pos: int
    sent_idx: int


class BiEncoderParams(TypedDict):
    max_context_length: int
    max_cand_length: int
    debug: bool
    eval_batch_size: int


class BiEncoderOutput(NamedTuple):
    labels: List[int]
    nearest_neighbors: List[np.ndarray]
    scores: List[np.ndarray]


class CrossEncoderDataPreparationOutput(NamedTuple):
    context_input: torch.LongTensor
    candidate_input: torch.LongTensor
    label_input: torch.LongTensor


class CrossEncoderParams(TypedDict):
    eval_batch_size: int


class CrossEncoderOutput(NamedTuple):
    predictions: np.ndarray
    logits: np.ndarray


def extract_mention_data(mention: Mention, sentences: List[str]) -> MentionAnnotation:
    """
    Extract and format data for a single mention.
    
    Args:
        mention (Mention): The mention data from the NER model output.
        sentences (List[str]): The list of sentences from the NER model output.

    Returns:
        MentionAnnotation: A dictionary containing the formatted mention data.
    """
    return MentionAnnotation(
        label="unknown",
        label_id=-1,
        context_left=sentences[mention["sent_idx"]][:mention["start_pos"]].lower(),
        context_right=sentences[mention["sent_idx"]][mention["end_pos"]:].lower(),
        mention=mention["text"].lower(),
        start_pos=int(mention["start_pos"]),
        end_pos=int(mention["end_pos"]),
        sent_idx=mention["sent_idx"]
    )


def annotate_mentions(ner_model: Any, input_sentences: List[str]) -> List[MentionAnnotation]:
    """
    Annotate named entity mentions in the input sentences using the provided NER model.

    Args:
        ner_model (Any): The named entity recognition model to use for prediction.
        input_sentences (List[str]): A list of input sentences to be annotated.

    Returns:
        List[MentionAnnotation]: A list of dictionaries containing annotated mention data.
    """
    # Predict named entities in the input sentences
    ner_output_data = ner_model.predict(input_sentences)
    sentences = ner_output_data["sentences"]
    mentions: List[Mention] = ner_output_data["mentions"]

    # Prepare the annotated samples
    annotated_samples: List[MentionAnnotation] = [
        extract_mention_data(mention, sentences) for mention in mentions
    ]
    
    return annotated_samples


def process_biencoder_dataloader(
    samples: List[MentionAnnotation],
    tokenizer: PreTrainedTokenizer,
    biencoder_params: BiEncoderParams
) -> DataLoader:
    """
    Processes samples to create a DataLoader for the biencoder.

    Args:
        samples (List[Dict[str, Any]]): The input samples for the dataloader.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for processing the samples.
        biencoder_params (BiEncoderParams): Parameters for the biencoder including context and candidate lengths.

    Returns:
        DataLoader: A DataLoader for the biencoder with the processed samples.
    """
    _, tensor_data = process_mention_data(
        samples,
        tokenizer,
        biencoder_params["max_context_length"],
        biencoder_params["max_cand_length"],
        silent=True,
        logger=None,
        debug=biencoder_params["debug"],
    )
    
    sequential_sampler = SequentialSampler(tensor_data)
    data_loader = DataLoader(
        tensor_data,
        sampler=sequential_sampler,
        batch_size=biencoder_params["eval_batch_size"]
    )
    
    return data_loader


def run_biencoder(
    biencoder: BiEncoderRanker,
    dataloader: DataLoader,
    candidate_encoding: Optional[torch.Tensor] = None,
    top_k: int = 100,
    indexer: Optional[DenseIndexer] = None
) -> BiEncoderOutput:
    """
    Run the biencoder model to get the top-k nearest neighbors and their scores.

    Args:
        biencoder (BiEncoderRanker): The biencoder model to use.
        dataloader (DataLoader): The dataloader for the input data.
        candidate_encoding (Optional[torch.Tensor]): Encodings of the candidate entities.
        top_k (int): Number of top nearest neighbors to return.
        indexer (Optional[DenseIndexer]): Optional dense indexer for searching nearest neighbors.

    Returns:
        BiEncoderOutput: Named tuple containing labels, nearest neighbors indices, and scores.
    """
    biencoder.model.eval()
    all_labels = []
    all_nearest_neighbors = []
    all_scores = []

    for batch in tqdm(dataloader):
        context_input, _, label_ids = batch
        context_input = context_input.to(device=biencoder.device)

        with torch.no_grad():
            if indexer is not None:
                context_encodings = biencoder.encode_context(context_input).numpy()
                context_encodings = np.ascontiguousarray(context_encodings)
                scores, indices = indexer.search_knn(context_encodings, top_k)
            else:
                scores = biencoder.score_candidate(
                    context_input.to(biencoder.device),
                    None,
                    cand_encs=candidate_encoding.to(biencoder.device)
                )
                scores, indices = scores.topk(top_k)
                scores = scores.cpu().data.numpy()
                indices = indices.cpu().data.numpy()

        all_labels.extend(label_ids.data.numpy())
        all_nearest_neighbors.extend(indices)
        all_scores.extend(scores)

    return BiEncoderOutput(labels=all_labels, nearest_neighbors=all_nearest_neighbors, scores=all_scores)


def prepare_crossencoder_data(
    tokenizer: PreTrainedTokenizer,
    samples: List[MentionAnnotation],
    labels: List[int],
    nns: List[List[int]],
    id2title: Dict[int, str],
    id2text: Dict[int, str],
    keep_all: bool = False
) -> CrossEncoderDataPreparationOutput:
    """
    Prepares data for crossencoder training by encoding mentions and candidates.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer to use for encoding.
        samples (List[Sample]): The input samples.
        labels (List[int]): The labels for the samples.
        nns (List[List[int]]): The nearest neighbors for each sample.
        id2title (Dict[int, str]): Mapping from IDs to entity titles.
        id2text (Dict[int, str]): Mapping from IDs to entity texts.
        keep_all (bool): Flag to keep all examples or filter based on gold entity presence.

    Returns:
        CrossEncoderOutput: Named tuple containing tensors for context input, candidate input, and label input.
    """

    # Encode mentions
    context_input_list = prepare_crossencoder_mentions(tokenizer, samples)

    # Encode candidates (output of biencoder)
    label_input_list, candidate_input_list = prepare_crossencoder_candidates(
        tokenizer, labels, nns, id2title, id2text
    )

    if not keep_all:
        # Remove examples where the gold entity is not among the candidates
        (
            context_input_list,
            label_input_list,
            candidate_input_list,
        ) = filter_crossencoder_tensor_input(
            context_input_list, label_input_list, candidate_input_list
        )
    else:
        label_input_list = [0] * len(label_input_list)

    context_input = torch.LongTensor(context_input_list)
    label_input = torch.LongTensor(label_input_list)
    candidate_input = torch.LongTensor(candidate_input_list)

    return CrossEncoderDataPreparationOutput(context_input, candidate_input, label_input)


def process_crossencoder_dataloader(
    context_input: torch.LongTensor,
    label_input: torch.LongTensor,
    crossencoder_params: CrossEncoderParams
) -> DataLoader:
    """
    Processes input tensors to create a DataLoader for the crossencoder.

    Args:
        context_input (torch.LongTensor): Tensor of context input sequences.
        label_input (torch.LongTensor): Tensor of label input sequences.
        crossencoder_params (CrossEncoderParams): Parameters for the crossencoder, including batch size.

    Returns:
        DataLoader: A DataLoader for the crossencoder with the processed input tensors.
    """
    tensor_data = TensorDataset(context_input, label_input)
    sampler = SequentialSampler(tensor_data)
    dataloader = DataLoader(
        tensor_data,
        sampler=sampler,
        batch_size=crossencoder_params["eval_batch_size"]
    )
    return dataloader


def run_crossencoder(
    crossencoder: CrossEncoderRanker,
    dataloader: DataLoader,
    context_len: int,
    device: str = "cuda"
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Runs the crossencoder model on the given dataloader and evaluates its performance.

    Args:
        crossencoder (CrossEncoderRanker): The crossencoder model to use.
        dataloader (DataLoader): The dataloader for input data.
        logger (Optional[Any]): Logger for logging information, can be None.
        context_len (int): The context length for the evaluation.
        device (str): The device to run the model on, default is "cuda".

    Returns:
        Tuple[float, np.ndarray, np.ndarray]: The accuracy, predictions, and logits of the crossencoder model.
    """
    crossencoder.model.eval()
    crossencoder.to(device)

    eval_accuracy = 0.0
    num_eval_examples = 0

    all_logits = []

    for step, batch in enumerate(tqdm(dataloader)):
        batch = tuple(t.to(device) for t in batch)
        context_input, label_input = batch[0], batch[1]

        with torch.no_grad():
            eval_loss, logits = crossencoder(context_input, label_input, context_len)

        logits = logits.detach().cpu().numpy()
        label_ids = label_input.cpu().numpy()

        all_logits.extend(logits)

        num_eval_examples += context_input.size(0)

    predictions = np.argsort(all_logits, axis=1)

    return predictions, np.array(all_logits)


import json
import jsonlines


def load_biencoder_ranker(model_path: str, config_path: str, logger: Optional[Logger]=None) -> BiEncoderRanker:
    # load biencoder model
    if logger:
        logger.info("loading biencoder model")
    try:
        with open(config_path) as json_file:
            biencoder_params = json.load(json_file)
    except json.decoder.JSONDecodeError:
        with open(config_path) as json_file:
            for line in json_file:
                line = (
                    line
                        .replace("'", "\"")
                        .replace("True", "true")
                        .replace("False", "false")
                        .replace("None", "null")
                )
                biencoder_params = json.loads(line)
                break

    biencoder_params["path_to_model"] = model_path
    biencoder = BiEncoderRanker(biencoder_params)
    return biencoder


def load_crossencoder_ranker(model_path: str, config_path: str, logger: Optional[Logger]=None) -> CrossEncoderRanker:
    # load crossencoder model
    if logger:
        logger.info("loading crossencoder model")
    try:
        with open(config_path) as json_file:
            crossencoder_params = json.load(json_file)
    except json.decoder.JSONDecodeError:
        with open(config_path) as json_file:
            for line in json_file:
                line = (
                    line
                        .replace("'", "\"")
                        .replace("True", "true")
                        .replace("False", "false")
                        .replace("None", "null")
                )
                crossencoder_params = json.loads(line)
                break
    crossencoder_params["path_to_model"] = model_path
    crossencoder = CrossEncoderRanker(crossencoder_params)
    return crossencoder


class DescriptionEntity(NamedTuple):
    title: str
    text: str
    wikipedia_url: str


def load_candidate_entities(
        entity_catalogue_path: str, 
        entity_description_embeddings_path: str, 
        logger: Optional[Logger]=None
    ) -> Tuple[torch.Tensor, Dict[str, int], Dict[int, str], Dict[int, str], Dict[str, int]]:
    candidate_encoding: torch.Tensor = torch.load(entity_description_embeddings_path)

    # load all the 5903527 entities
    title2id: Dict[str, int] = {}
    id2title: Dict[int, str] = {}
    id2text: Dict[int, str] = {}
    wikipedia_id2local_id: Dict[str, int] = {}

    with jsonlines.open(entity_catalogue_path, "r") as reader:
        for local_idx, entity in enumerate(reader):
            if "idx" in entity:
                split = entity["idx"].split("curid=")
                if len(split) > 1:
                    wikipedia_id = int(split[-1].strip())
                else:
                    wikipedia_id = entity["idx"].strip()

                assert wikipedia_id not in wikipedia_id2local_id
                wikipedia_id2local_id[wikipedia_id] = local_idx

            title2id[entity["title"]] = local_idx
            id2title[local_idx] = entity["title"]
            id2text[local_idx] = entity["text"]

    return (
        candidate_encoding,
        title2id,
        id2title,
        id2text,
        wikipedia_id2local_id,
    )
