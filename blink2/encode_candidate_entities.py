from transformers import pipeline
from dataclasses import dataclass
from typing import Any, List
import jsonlines
from datasets import Dataset


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
        yield row


def main():

    entity_encoder = pipeline(
        "entity-encoder", 
        model='shomez/blink-biencoder-description-encoder', 
        device=0, 
        batch_size=64
    )

    entities = read_entities('/content/all_entities.jsonl')

    entities_ds = Dataset.from_generator(ds_gen, gen_kwargs={'items': entities})

    entities_embeddings = entity_encoder(
        entities_ds, 
        return_tensors=True, 
        tokenize_kwargs={
            'truncation': True, 
            'max_length': 128
        }
    )

    entities_ds = entities_ds.add_column('embeddings', column=entities_embeddings)
    entities_ds.add_faiss_index(column='embeddings')


