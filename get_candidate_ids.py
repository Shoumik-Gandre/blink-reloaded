from transformers import AutoTokenizer
from datasets import Dataset, load_dataset

entities: Dataset = load_dataset("json", data_files='models/entity.jsonl', split='train')

tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-mini')
def tokenize(example, tokenizer): return tokenizer(example['title'], example['text'], truncation='longest_first', max_length=512)

entities_tk = entities.map(tokenize, fn_kwargs={'tokenizer': tokenizer}, num_proc=4)

def correct_entity_token(example, sep_token_id, entity_sep_token_id):
    input_ids: list = example['input_ids']
    sep_token_idx = input_ids.index(sep_token_id)
    input_ids[sep_token_idx] = entity_sep_token_id
    example['input_ids'] = input_ids
    return example
    
entities_tk2 = (
    entities_tk
    .map(
        correct_entity_token, 
        fn_kwargs={
            'sep_token_id': tokenizer.sep_token_id,
            'entity_sep_token_id': tokenizer.convert_tokens_to_ids("[unused2]")
        }, 
        num_proc=4
    )
)
