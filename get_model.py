from pathlib import Path
from safetensors.torch import load_file
from blink.biencoder.biencoder import BiEncoderRanker

params = {'path_to_model': r'D:\projects\blink-reloaded\models\zeshel\biencoder2\pytorch_model.bin', 'silent': False, 'debug': False, 'data_parallel': False, 'no_cuda': False, 'top_k': 10, 'seed': 52313, 'zeshel': False, 'max_seq_length': 256, 'max_context_length': 128, 'max_cand_length': 128, 'path_to_model': None, 'bert_model': 'prajjwal1/bert-mini', 'pull_from_layer': -1, 'lowercase': True, 'context_key': 'context', 'out_dim': 1, 'add_linear': False, 'data_path': 'data/zeshel/blink_format', 'output_path': 'models/zeshel/biencoder2', 'evaluate': False, 'output_eval_file': None, 'train_batch_size': 64, 'max_grad_norm': 1.0, 'learning_rate': 2e-05, 'num_train_epochs': 10, 'print_interval': 10, 'eval_interval': 1000, 'save_interval': 1, 'warmup_proportion': 0.1, 'gradient_accumulation_steps': 1, 'type_optimization': 'all_encoder_layers', 'shuffle': False, 'eval_batch_size': 64, 'mode': 'valid', 'save_topk_result': False, 'encode_batch_size': 8, 'cand_pool_path': None, 'entity_dict_path': None, 'cand_encode_path': None, 'roberta': False}

model = BiEncoderRanker(params)
state_dict = load_file(r'models\zeshel\biencoder3\checkpoint-1155\model.safetensors')
model.load_state_dict(state_dict)

model.model.cand_encoder.bert_model.save_pretrained(str(Path(r"models\zeshel\biencoder3") / "entity-encoder1"))
model.model.context_encoder.bert_model.save_pretrained(str(Path(r"models\zeshel\biencoder3") / "mention-encoder1"))