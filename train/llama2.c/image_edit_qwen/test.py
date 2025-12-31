import os, time, sys
import torch
from PIL import Image
sys.path.insert(0, '/home/user/.bin/learn/train/data/diffusers/src/')
from diffusers import QwenImageEditPlusPipeline
#import pdb; pdb.set_trace() 
torch.set_num_threads(18)

model_path = "Qwen-Image-Edit-2511/"
pipeline = QwenImageEditPlusPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
print("pipeline loaded")

pipeline.to('cpu')
pipeline.set_progress_bar_config(disable=None)
image1 = Image.open("input1.png")
image2 = Image.open("input2.png")
prompt = "将图1中女生的衣服替换为图2中女生的衣服"
inputs = {
    "image": [image1, image2],
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 1,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
}
with torch.inference_mode():
    start = time.time()
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_image.save("out.jpg")
    print('spend', time.time() - start)
