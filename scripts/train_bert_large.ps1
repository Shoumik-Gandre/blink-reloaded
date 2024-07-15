$env:PYTHONPATH = ".."
python ../blink2/biencoder/train_biencoder_bert.py `
--data-path shomez/zeshel-blink `
--model-name google-bert/bert-large-uncased `
--output-path models/zeshel/bert-large-uncased-300 `
--learning-rate 2e-05 `
--num-train-epochs 5 `
--max-context-length 128 `
--max-cand-length 128 `
--train-batch-size 8 `
--embed-dim 300 `
--debug