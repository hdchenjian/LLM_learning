#python ddp_pretrain.py --data_path ../../data/llm/tiny_monkey_general_open_corpus.jsonl --gpus 0 --batch_size 1 > .aa
#python ddp_sft_full.py --data_path ../../data/llm/tiny_BelleGroup_sft.jsonl --gpus 0 --batch_size 16 > .bb &

#cpu train
export OMP_NUM_THREADS=1
python ddp_pretrain.py --data_path ../../data/llm/tiny_monkey_general_open_corpus.json --device cpu --batch_size 1
