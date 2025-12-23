from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch, time

model_path = '/home/user/.bin/learn/train/data/llm/qwen3-0.6b'
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

#model = AutoPeftModelForCausalLM.from_pretrained("out")
model = AutoPeftModelForCausalLM.from_pretrained("Qwen3_lora/checkpoint-57/")
model = model.to("cuda")
model.eval()

prompt = "龙琳琳   ，宁夏回族自治区璐市城东林街g座 955491，nafan@example.com。小区垃圾堆积成山，晚上噪音扰人清梦，停车难上加难，简直无法忍受！太插件了阿萨德"
messages = [{"role": "system", "content": "将文本中的name、address、email、question提取出来，以json格式输出，字段为name、address、email、question，值为文本中提取出来的内容。"},
            {"role": "user", "content": prompt}
            ]

inputs = tokenizer.apply_chat_template(messages,
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True,
                                       enable_thinking=False).to('cuda')

gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
with torch.no_grad():
    for i in range(1):
        start = time.time()
        outputs = model.generate(**inputs, **gen_kwargs)
        print(i, 'spend', time.time() - start)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
