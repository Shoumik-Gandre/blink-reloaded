"""
$env:PYTHONPATH='.'
python .\blink2\encode_candidate_entities.py models\zeshel\biencoder3\epoch_2\entity-encoder models\zeshel\biencoder3\epoch_2\ data\all_entities.jsonl data\biencoder3
"""
from typing import Annotated, Any, List, TypedDict

import typer
import jsonlines
import torch
from datasets import Dataset, load_dataset
from transformers import pipeline

from blink2.biencoder import register_entity_encoder_pipeline

register_entity_encoder_pipeline()


# @dataclass
class Entity(TypedDict):
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

# device = 0 if torch.cuda.is_available() else -1

# entity_encoder = pipeline(
#     "entity-encoder", 
#     model=r'D:\projects\blink-reloaded\models\zeshel\entity-encoder-bertmini',
#     device=device, 
#     batch_size=128
# )

# entities = read_entities('data/all_entities.jsonl')

# entities_embeddings = entity_encoder(
#     entities, 
#     tokenize_kwargs={
#         'truncation': True, 
#         'max_length': 128
#     }
# )

# entities_ds: Dataset = Dataset.from_generator(ds_gen, gen_kwargs={'items': entities})
# entities_ds = entities_ds.add_column('embeddings', column=entities_embeddings)
# entities_ds.save_to_disk("data/test2")
# entities_ds.add_faiss_index(column='embeddings', metric_type=faiss.METRIC_INNER_PRODUCT)
# entities_ds.save_faiss_index('embeddings', 'data/all_entities.faiss')

def main(
    model_path: Annotated[str, typer.Argument(help="Entity Embedder")],
    tokenizer_path: Annotated[str, typer.Argument(help="Tokenizer for Entity Embedder")],
    entities_path: Annotated[str, typer.Argument(help="Path to entities.jsonl file")],
    output_path: Annotated[str, typer.Argument(help="Path to save produced entity embedding dataset")],
) -> None:
    
    device = 0 if torch.cuda.is_available() else -1

    entity_encoder = pipeline(
        "entity-encoder", 
        model=model_path,
        tokenizer=tokenizer_path,
        device=device, 
        batch_size=128
    )
    entities_ds = load_dataset("json", data_files=entities_path, split='train')
    print(entities_ds)
    # entities = read_entities(entities_path)
    entities_embeddings = entity_encoder(
        entities_ds.to_list(), 
        tokenize_kwargs={
            'truncation': True, 
            'max_length': 128
        }
    )
    print(len(entities_embeddings))
    # entities_ds: Dataset = Dataset.from_generator(ds_gen, gen_kwargs={'items': entities})
    entities_ds = entities_ds.add_column('embeddings', column=entities_embeddings)
    entities_ds.save_to_disk(output_path)


if __name__ == '__main__':
    app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)
    app.command()(main)
    app()