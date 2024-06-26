from transformers import pipeline
from dataclasses import dataclass, asdict
from typing import Any, List
import jsonlines
from datasets import Dataset
import faiss
import torch

from blink2.biencoder import register_entity_encoder_pipeline

register_entity_encoder_pipeline()


@dataclass
class Entity:
    text: str
    idx: str
    title: str
    entity: str


def read_entities(file_path: str) -> List[Entity]:
    entities = []
    with jsonlines.open(file_path) as reader:
        for entity in reader:
            entities.append(Entity(**entity))
    return entities


def ds_gen(items: List[Any]):
    for row in items:
        yield asdict(row)

device = 0 if torch.cuda.is_available() else -1

entity_encoder = pipeline(
    "entity-encoder", 
    model='shomez/blink-biencoder-description-encoder', 
    device=device, 
    batch_size=64
)

entities = read_entities('/content/all_entities.jsonl')


entities_embeddings = entity_encoder(
    entities, 
    tokenize_kwargs={
        'truncation': True, 
        'max_length': 128
    }
)

entities_ds: Dataset = Dataset.from_generator(ds_gen, gen_kwargs={'items': entities})
entities_ds = entities_ds.add_column('embeddings', column=entities_embeddings)
entities_ds.add_faiss_index(column='embeddings', metric_type=faiss.METRIC_INNER_PRODUCT)
entities_ds.save_faiss_index('embeddings', 'all_entities.faiss')
