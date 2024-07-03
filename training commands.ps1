$env:PYTHONPATH = "."
python blink/biencoder/train_biencoder.py `
  --data_path data/zeshel/blink_format `
  --output_path models/zeshel/biencoder-test `
  --learning_rate 2e-05 --num_train_epochs 5 --max_context_length 128 --max_cand_length 128 `
  --train_batch_size 16 --eval_batch_size 16 --eval_interval 10000 --bert_model prajjwal1/bert-mini `
  --type_optimization all_encoder_layers --debug


$env:PYTHONPATH = "."
python blink/biencoder/train_biencoder.py `
--data_path data/zeshel/blink_format `
--output_path models/zeshel/biencoder3 `
--bert_model prajjwal1/bert-mini `
--learning_rate 2e-05 `
--num_train_epochs 10 `
--max_context_length 128 `
--max_cand_length 128 `
--train_batch_size 64 `
--eval_batch_size 64 `
--eval_interval 1000 `
--type_optimization all_encoder_layers `
--shuffle true `
--zeshel true

$env:PYTHONPATH = "."
python blink/biencoder/train_biencoder.py `
--data-path data/zeshel/blink_format `
--output-path models/zeshel/biencoder3 `
--bert-model prajjwal1/bert-mini `
--learning-rate 1e-05 `
--num-train-epochs 10 `
--max-context-length 128 `
--max-cand-length 128 `
--train-batch-size 64 `
--eval-batch-size 64 `
--eval-interval 100000 `
--shuffle `
--zeshel `
--debug
