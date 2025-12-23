from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
from peft import LoraConfig, TaskType, get_peft_model
import torch
import swanlab
from swanlab.integration.transformers import SwanLabCallback

# swanlab token: 2qwSxNgtQVm7a3tJi5CL2
# https://datawhalechina.github.io/base-nlp/#/chapter11/02_lora

def process_func(example):
    MAX_LENGTH = 1024 # 设置最大序列长度为1024个token
    input_ids, attention_mask, labels = [], [], [] # 初始化返回值
    # 适配chat_template
    instruction = tokenizer(
        f"<|im_start|>system\n{example['system']}<|im_end|>\n"
        f"<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n",
        add_special_tokens=False
    )
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    # 将instructio部分和response部分的input_ids拼接，并在末尾添加eos token作为标记结束的token
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    # 注意力掩码，表示模型需要关注的位置
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    # 对于instruction，使用-100表示这些位置不计算loss（即模型不需要预测这部分）,  最后一个为结束符
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 超出最大序列长度截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# 将JSON文件转换为CSV文件
df = pd.read_json('fake_sft.json')
ds = Dataset.from_pandas(df)
#model_id = "Qwen/Qwen3-0.6B"
model_path = '/home/user/.bin/learn/train/data/llm/qwen3-0.6b'
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
#print('tokenizer', tokenizer)

tokenized_id = ds.map(process_func, remove_columns=ds.column_names)
print('tokenized_id', type(tokenized_id), len(tokenized_id))
#print(tokenizer.decode(tokenized_id[0]['input_ids']), '\n')
#print(tokenized_id[0]["labels"], '\ndecode:', list(filter(lambda x: x != -100, tokenized_id[0]["labels"])))
#print(tokenizer.decode(list(filter(lambda x: x != -100, tokenized_id[0]["labels"]))))
#print('tokenized_id[0]["labels"]', tokenized_id[0])
#print(tokenizer.decode(tokenized_id[0]["input_ids"]))
#exit()
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", dtype=torch.bfloat16)
model.enable_input_require_grads() # 开启梯度检查点时的必要设置[配合gradient_checkpointing参数]，通过在前向传播时释放中间激活值，并在反向传播时重新计算这些值来节省显存
n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
print(f"继承一个预训练模型 - Total size={n_params/2**20:.2f}M params")


config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩, LoRA 的旁路结构在训练完成后，可以通过矩阵加法直接“合并”回原始权重中,不会引入额外的计算
    lora_alpha=32, # 可调缩放超参, 与 r 一起决定了 LoRA 更新的强度。实际缩放比例为: s = lora_alpha/r, O = [W  + s *(B * A)]  * x
    lora_dropout=0.1# Dropout 比例
)
model = get_peft_model(model, config)
model.print_trainable_parameters()  # 模型参数训练量只有0.8395%

args = TrainingArguments(
    output_dir="Qwen3_lora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=1,
    num_train_epochs=3,
    save_steps=50,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)

# 实例化SwanLabCallback
swanlab_callback = SwanLabCallback(project="Qwen3-Lora", experiment_name="Qwen3-0.6B-extarct-lora-2")
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback]
)
trainer.train()
model.save_pretrained("out")
