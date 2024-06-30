$env:PYTHONPATH = "." 
python blink/biencoder/eval_biencoder.py `
--data_path DATA_PATH `
--save_topk_result `
--path_to_model models/zeshel/biencoder/pytorch_model.bin `
--mode test `
--eval_batch_size 16 `
--output_path ./result/top_k  `
--max_context_length 128 `
--max_cand_length 128 `
--bert_model prajjwal1/bert-mini `
--entity_dict_path data/zeshel/documents `
--top_k 64 `
--zeshel true `
--cand_encode_path encode.t7 `
--cand_pool_path pool.t7 `
--debug

$env:PYTHONPATH = "." 
python blink/biencoder/eval_biencoder.py `
--data_path DATA_PATH `
--save_topk_result `
--path_to_model models/zeshel/biencoder/pytorch_model.bin `
--mode test `
--eval_batch_size 64 `
--output_path ./result/top_k  `
--max_context_length 128 `
--max_cand_length 128 `
--bert_model prajjwal1/bert-mini `
--entity_dict_path data/zeshel/documents `
--top_k 64 `
--zeshel true `
--cand_encode_path models/encode.t7 `
--cand_pool_path models/pool.t7


$env:PYTHONPATH = "." 
python blink/biencoder/eval_biencoder.py `
--data_path DATA_PATH `
--save_topk_result `
--path_to_model models/biencoder_wiki_large.bin `
--mode test `
--eval_batch_size 64 `
--output_path ./result/top_k  `
--max_context_length 128 `
--max_cand_length 128 `
--bert_model bert-large-uncased `
--entity_dict_path data/zeshel/documents `
--top_k 64 `
--zeshel true `
--cand_encode_path models/encode2.t7 `
--cand_pool_path models/pool2.t7