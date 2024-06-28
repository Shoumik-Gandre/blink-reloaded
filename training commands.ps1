$env:PYTHONPATH = "."
python blink/biencoder/train_biencoder.py `
  --data_path data/zeshel/blink_format `
  --output_path models/zeshel/biencoder `  
  --learning_rate 2e-05 --num_train_epochs 5 --max_context_length 128 --max_cand_length 128 `
  --train_batch_size 16 --eval_batch_size 16 --bert_model bert-base-uncased `
  --type_optimization all_encoder_layers --data_parallel