from enum import Enum
import logging
from flask import Flask, request, jsonify, request
from pathlib import Path
import blink.ner as NER
import blink.main_dense as main_dense
from typing import Dict, List, NamedTuple, Set
import jsonlines
from .main_utils import (
    MentionAnnotation, 
    process_biencoder_dataloader, 
    run_biencoder, 
    annotate_mentions, 
    prepare_crossencoder_data,
    process_crossencoder_dataloader,
    load_candidate_entities,
    load_biencoder_ranker,
    load_crossencoder_ranker
)

# Initialize Flask app
app = Flask(__name__)

# Configure paths and load models
models_path = Path() / "models" 
entity_sets = Path() / "entity_sets"

config = {
    "biencoder_model": models_path / "biencoder_wiki_large.bin",
    "biencoder_config": models_path / "biencoder_wiki_large.json",
    "crossencoder_model": models_path / "crossencoder_wiki_large.bin",
    "crossencoder_config": models_path / "crossencoder_wiki_large.json",
    "entity_catalogue": entity_sets / "all_entities.jsonl",
    "entity_encoding": Path() / "entity_embeddings/0_-1.t7",
    "top_k": 3,
}

# Load Models
biencoder = load_biencoder_ranker(config["biencoder_model"], config['biencoder_config'])
crossencoder = load_crossencoder_ranker(config["crossencoder_model"], config['crossencoder_config'])
(
    candidate_encoding,
    title2id,
    id2title,
    id2text,
    wikipedia_id2local_id,
) = load_candidate_entities(config['entity_catalogue'], config['entity_encoding'])
ner_model = NER.get_model()

class EntityPrediction(NamedTuple):
    label: str
    score: float


def get_predictions(samples: List[MentionAnnotation], top_k: int=5) -> List[EntityPrediction]:
    biencoder_dataloader = process_biencoder_dataloader(samples, biencoder.tokenizer, biencoder.params)
    biencoder_output = run_biencoder(biencoder, biencoder_dataloader, candidate_encoding, top_k, None)

    context_input, candidate_input, label_input = prepare_crossencoder_data(crossencoder.tokenizer, samples, biencoder_output.labels, biencoder_output.nearest_neighbors, id2title, id2text, True)
    context_input = main_dense.modify(context_input, candidate_input, crossencoder.params["max_seq_length"])
    dataloader = process_crossencoder_dataloader(context_input, label_input, crossencoder.params)

    accuracy, index_array, unsorted_scores = main_dense._run_crossencoder(crossencoder, dataloader, None, context_len=biencoder.params["max_context_length"], device='cpu')

    entities: List[EntityPrediction] = []
    for entity_list, index_list, scores_list in zip(biencoder_output.nearest_neighbors, index_array, unsorted_scores):
        index_list = index_list.tolist()
        index_list.reverse()

        for index in index_list:
            e_id = entity_list[index]
            e_title = id2title[e_id]
            e_score = scores_list[index]
            entity = EntityPrediction(label=e_title, score=e_score)
            entities.append(entity)

    return entities


def load_entity_set(entity_set_path) -> Set[str]:
    entity_set = set()
    with jsonlines.open(entity_set_path) as reader:
        for obj in reader:
            entity_set.add(obj['title'])
    return entity_set


class Topics(Enum):
    GUN_CONTROL = 'gun-control'
    ABORTION = 'abortion'

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_ 


# Example topics
TOPIC_TOGGLE = {
    Topics.GUN_CONTROL: False,
    Topics.ABORTION: False,
}


TOPIC_ENTITY_SETS = {
    Topics.GUN_CONTROL: load_entity_set(entity_sets / "gun_entities.jsonl"),
    Topics.ABORTION: load_entity_set(entity_sets / "abortion_entities.jsonl"),
}


@app.route("/", methods=["GET", "POST"])
def index():
    return "Entity Linking Topic Detector."


@app.route('/predict', methods=['POST'])
def predict() -> Dict[str, str]:
    data = request.get_json()
    sentence = data.get('sentence', '')
    topic_set = data.get('topics', [])

    if not sentence:
        return jsonify({"error": "No sentence provided"}), 400
    
    topics_not_found = []
    for topic in topic_set:
        if not Topics.has_value(topic):
            topics_not_found.append(topic)
    if len(topics_not_found):
        return jsonify({"error": f"Topics not found {topics_not_found}"}), 400
    del topics_not_found
    
    if not len(topic_set):
        return jsonify({
            "prediction": {}
        }), 200

    samples = annotate_mentions(ner_model, [sentence])
    entities = get_predictions(samples, top_k=config['top_k'])

    # Remove all candidates with a score less than or equal to 0
    entities = [entity for entity in entities if entity.score > 0]

    topic_detection = {
        topic.value: False
        for topic in Topics
        if topic.value in topic_set
    }
    for entity in entities:
        for topic in Topics:
            if topic.value in topic_set and entity.label in TOPIC_ENTITY_SETS[topic]:
                topic_detection[topic.value] = True
    
    return jsonify({
        "prediction": topic_detection,
    }), 200


if __name__ == "__main__":
    app.run(debug=True)
else:
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)