import os
import pickle
from contextlib import nullcontext
import torch
from k_model import ModelConfig, Transformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from model_sample import TextGenerator

generator = TextGenerator(checkpoint='./sft_model_215M/sft_dim1024_layers18_vocab_size6144.pth')
tokenizer = generator.tokenizer
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device_type = 'cuda' if 'cuda' in device else 'cpu'
dtype="bfloat16"
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


while True:
    prompt = input('User: ')
    messages = [{"role": "system", "content": "你是一个AI助手"}]
    messages.append({"role": "user", "content": prompt})

    input_ids = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(input_ids).data['input_ids']
    x = (torch.tensor(input_ids, dtype=torch.long)[None, ...]).to(device)

    with torch.no_grad():
        with ctx:  # 进入自动混合精度的上下文（如果是 GPU 并使用 float16 时）
            y = generator.model.generate(x, stop_id=tokenizer.eos_token_id, max_new_tokens=512, temperature=0.6)
            response = tokenizer.decode(y[0].tolist())
    print(f"Assistant: {response}")
    #messages.append({"role": "assistant", "content": response})
