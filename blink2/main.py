from typing import List, Dict, Any, Optional
from flair.data import Sentence
from flair.models import SequenceTagger
from transformers import pipeline
from datasets import load_dataset


class Flair:
    def __init__(self, parameters: Optional[Dict[str, Any]] = None) -> None:
        self.model = SequenceTagger.load("ner")

    def predict(self, sentences: List[str]) -> Dict[str, Any]:
        mentions = []
        for sent_idx, sent in enumerate(sentences):
            sentence = Sentence(sent, use_tokenizer=True)
            self.model.predict(sentence)
            sent_mentions = sentence.to_dict(tag_type="ner")["entities"]
            for mention in sent_mentions:
                mention["sent_idx"] = sent_idx
            mentions.extend(sent_mentions)
        return {"sentences": sentences, "mentions": mentions}


def annotate_ner(ner_model: Flair, input_sentences: List[str]) -> List[Dict[str, Any]]:
    ner_output_data = ner_model.predict(input_sentences)
    sentences = ner_output_data["sentences"]
    mentions = ner_output_data["mentions"]
    samples = []

    for mention in mentions:
        record = {
            "context_left": sentences[mention["sent_idx"]][:mention["start_pos"]].lower(),
            "context_right": sentences[mention["sent_idx"]][mention["end_pos"] :].lower(),
            "mention": mention["text"].lower(),
            "start_pos": int(mention["start_pos"]),
            "end_pos": int(mention["end_pos"]),
            "sent_idx": mention["sent_idx"]
        }
        samples.append(record)
    
    return samples


def main():
    ner = Flair()
    sentences = [
        "The overturing of Roe Wade was dissapointing.",
        "The Sandy Hook School Shooting gave rise to the March for our Lives."
    ]
    mentions = annotate_ner(ner, sentences)
    mention_encoder = pipeline(
        "mention-encoder",
        model='shomez/blink-biencoder-mention-encoder',
        device=-1,
        batch_size=64,
        tokenize_kwargs={
            'max_length': 128,
        },
    )
    mention_embeddings = mention_encoder(mentions)
    import numpy as np
    entities_ds = load_dataset('shomez/gun-control-abortion-entities')
    entities_ds.add_faiss_index()

    result = entities_ds.search_batch('embeddings', np.array(mention_embeddings), k=3)
    for mention_index in result.total_indices.tolist():
        for candidate in mention_index:
            print(entities_ds[candidate]['title'])
        print()


if __name__ == '__main__':
    main()