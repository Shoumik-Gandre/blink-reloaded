from pathlib import Path
from blink.biencoder.biencoder2 import DebertaV2BiencoderRankerConfig, DebertaV2BiencoderRanker


checkpoint_path = Path(r'D:\projects\blink-reloaded\models\zeshel\deberta-v3-xsmall\checkpoint-10')
config = DebertaV2BiencoderRankerConfig.from_pretrained(checkpoint_path / 'config.json')

model = DebertaV2BiencoderRanker.from_pretrained(checkpoint_path)