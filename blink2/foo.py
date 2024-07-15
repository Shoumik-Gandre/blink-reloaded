from blink2.biencoder import MentionEncoderPoolPipeline

from pathlib import Path

from transformers import AutoTokenizer

from blink2.biencoder import DebertaV2BiencoderRanker, DebertaV2ForTextEncoding


sentences = [
    {
        "mention": "Roe V Wade",
        "context_left": "The overturing of",
        "context_right": "was dissapointing."
    },
    {
        "mention": "Sandy Hook School Elementary Shooting",
        "context_left": "The",
        "context_right": "gave rise to the"
    },
    {
        "mention": "March for our Lives",
        "context_left": "The Sandy Hook School Elementary Shooting gave rise to the",
        "context_right": "."
    }
]


base_dir = Path('/content/models/zeshel/deberta-v3-base/')
mention_encoder_path = base_dir / 'mention-encoder'
entity_encoder_path = base_dir / 'entity-encoder'

tokenizer = AutoTokenizer.from_pretrained(base_dir / 'checkpoint-10')
model = DebertaV2BiencoderRanker.from_pretrained(base_dir / 'checkpoint-10')

tokenizer.save_pretrained(mention_encoder_path)
model.mention_encoder.save_pretrained(mention_encoder_path)

tokenizer.save_pretrained(entity_encoder_path)
model.entity_encoder.save_pretrained(entity_encoder_path)

mention_encoder_model = DebertaV2ForTextEncoding.from_pretrained(mention_encoder_path)
mention_encoder_tokenizer = AutoTokenizer.from_pretrained(mention_encoder_path)

mention_encoder = MentionEncoderPoolPipeline(
    mention_encoder_model, 
    mention_encoder_tokenizer,
    device=-1,
    batch_size=64,
    tokenize_kwargs={
        'max_length': 128,
        'truncation': True,
    },
)

mention_encoder(sentences, **{'max_length': 128, 'truncation': True})