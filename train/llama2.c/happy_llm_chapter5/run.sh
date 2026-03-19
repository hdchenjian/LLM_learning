#python ddp_pretrain.py --data_path ../../data/llm/tiny_monkey_general_open_corpus.jsonl --gpus 0 --batch_size 1 > .aa
#python ddp_sft_full.py --data_path ../../data/llm/tiny_BelleGroup_sft.jsonl --gpus 0 --batch_size 16 > .bb &

#cpu train
export OMP_NUM_THREADS=1
#python ddp_pretrain.py --data_path ../../data/llm/tiny_monkey_general_open_corpus.json --device cpu --batch_size 1
#python ddp_sft_full.py --data_path ../../data/llm/tiny_BelleGroup_sft.jsonl --device cpu --batch_size 1
#modelscope download --model kmno4zx/happy-llm-215M-base pretrain_1024_18_6144.pth --local_dir ./dir
python ddp_sft_full.py --data_path aa --device cpu --batch_size 1
