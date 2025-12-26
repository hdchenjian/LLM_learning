import time, os, sys, torch, pickle
import numpy as np
sys.path.insert(0, '/home/user/.bin/learn/train/data/transformers/src/')
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
#torch.set_num_threads(8)
torch.set_grad_enabled(False)
#os.environ["OMP_NUM_THREADS"] = '1'
#import pdb; pdb.set_trace()

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model_path = "/home/user/.bin/learn/train/data/llm/qwen3-vl-2b"
processor = AutoProcessor.from_pretrained(model_path)

video = 0
if video:
    messages = [{
        "role": "user",
        "content": [{"type": "video", "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4",},
                    {"type": "text", "text": "描述这个视频"},
                    ],
    }]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        fps=0.02
    )
    print('video_grid_thw', inputs['video_grid_thw'].numpy())
elif video == -1:
    messages = [{
            "role": "user",
            "content": [{"type": "image", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",},
                        {"type": "text", "text": "描述这张图片"},
                        ],
        }]
    messages = [{
            "role": "user",
            "content": [{"type": "image", "image": "https://pic.616pic.com/ys_bnew_img/00/03/83/Nl6kJaIG4X.jpg",},
                        {"type": "image", "image": "http://img.keaiming.com/uploads/allimg/2020031310/uu30cagksun.jpeg",},
                        {"type": "text", "text": "简单描述这两张图片的共同点"},
                        ],
        }]
    
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    print('image_grid_thw', inputs['image_grid_thw'].numpy())
print('inputs', processor.apply_chat_template, processor.batch_decode(inputs['input_ids']))
#print('attention_mask', np.sum(inputs['attention_mask'].numpy()))
if 0:
    pickle.dump(inputs, open('inputs', 'wb'))
else:
    inputs = pickle.load(open('inputs', 'rb'))
for k in inputs:
    print(k, inputs[k].shape)
    if '_grid_thw' in k: print(k, inputs[k].numpy())
#exit()
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
#    attn_implementation="flash_attention_2",
    #device_map="auto",
    device_map="cpu",
)
model.eval()
inputs = inputs.to(model.device)

start = time.time()
generated_ids = model.generate(**inputs, max_new_tokens=256 / 32)
#import pdb; pdb.set_trace()
generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(output_text)
print('spend', time.time() - start)
