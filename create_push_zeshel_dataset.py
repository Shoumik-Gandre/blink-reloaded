from datasets import load_dataset

data_files = {
    'train': r'data\zeshel\blink_format\train.jsonl',
    'validation': r'data\zeshel\blink_format\valid.jsonl',
    'test': r'data\zeshel\blink_format\test.jsonl',
}

dataset = load_dataset("json", data_files=data_files)

dataset.push_to_hub('zeshel-blink')