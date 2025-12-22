#python ddp_pretrain.py --data_path ../../../../data/llm/tiny_monkey_general_open_corpus.jsonl --gpus 0 --batch_size 16 > .aa &
python ddp_sft_full.py --data_path ../../../../data/llm/tiny_monkey_general_open_corpus.jsonl --gpus 0 --batch_size 16 > .bb &
