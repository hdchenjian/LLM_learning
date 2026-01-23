# evalscope[perf]
if [ 0 -ge 1 ]; then
    #modelscope download --dataset modelscope/gsm8k --local_dir ./data/gsm8k
    #vllm bench throughput --help
    evalscope eval \
              --model qwen2.5 \
              --api-url http://localhost:8001/v1 --api-key EMPTY --eval-type openai_api \
              --datasets gsm8k \
              --dataset-args '{"gsm8k": {"local_path": "./data/gsm8k"}}' \
              --limit 5
elif [ 0 -ge 1 ]; then
    evalscope perf \
              --parallel 1 10 50 \
              --number 10 20 100 \
              --model qwen3-0.6b \
              --url http://localhost:8001/v1/chat/completions \
              --api openai \
              --dataset random \
              --max-tokens 1024 \
              --min-tokens 1024 \
              --prefix-length 0 \
              --min-prompt-length 1024 \
              --max-prompt-length 1024 \
              --tokenizer-path /home/luyao/d/data/llm/qwen/qwen3-0.6b \
              --extra-args '{"ignore_eos": true}'

else
    # Qwen/Qwen3-0.6B
    # V100 GPU does't support bfloat16
    #vllm serve "qwen3-0.6b/" --port 8001 --dtype bfloat16  --max-model-len 512 --gpu-memory-utilization 0.74
    #vllm serve "facebook_opt-125m" --port 8001 --dtype float16 --max-model-len 128 --cpu-offload-gb 10 --max_num_seqs=16 --gpu-memory-utilization 0.7
    python -m vllm.entrypoints.openai.api_server --model /home/luyao/d/data/llm/qwen/qwen3-8b --port 8001 --dtype bfloat16 --max-model-len 2048 --cpu-offload-gb 10 --max_num_seqs=128 --gpu-memory-utilization 0.98 --served-model-name qwen3-8b --trust-remote-code
    exit
    python -m vllm.entrypoints.openai.api_server --model /home/luyao/d/data/llm/qwen/qwen3-0.6b --port 8001 --dtype bfloat16 --max-model-len 2048 --cpu-offload-gb 10 --max_num_seqs=128 --gpu-memory-utilization 0.95 --served-model-name qwen3-0.6b --trust-remote-code
    exit
    python -m vllm.entrypoints.openai.api_server --model qwen2.5-0.5B-Instruct --port 8001 --dtype bfloat16 --max-model-len 1024 --cpu-offload-gb 10 --max_num_seqs=16 --gpu-memory-utilization 0.95 --served-model-name qwen2.5 --trust-remote-code
    #vllm serve "qwen2.5-0.5B-Instruct" --port 8001 --dtype bfloat16 --max-model-len 128 --cpu-offload-gb 10 --max_num_seqs=16 --gpu-memory-utilization 0.73
fi
