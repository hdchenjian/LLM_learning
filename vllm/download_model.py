import os

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 下载模型
#os.system('huggingface-cli download --resume-download facebook/opt-125m --local-dir autodl-tmp/')
#os.system('huggingface-cli download --resume-download Qwen/Qwen3-0.6B --local-dir autodl-tmp/')
os.system('huggingface-cli download --resume-download Qwen/Qwen2.5-0.5B-Instruct  --local-dir qwen2.5-0.5B-Instruct')
