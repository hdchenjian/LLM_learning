import os, time, sys
import torch
from PIL import Image
sys.path.insert(0, '/home/user/.bin/learn/train/data/diffusers/src/')
from diffusers import QwenImageLayeredPipeline
#import pdb; pdb.set_trace() 
#torch.set_num_threads(18)
start = time.time()

model_path = "Qwen-Image-Layered/"
pipeline = QwenImageLayeredPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
print("pipeline loaded")

pipeline.to('cpu')
image = Image.open("input1.png").convert("RGBA")
inputs = {
    "image": image,
    "generator": torch.Generator(device='cpu').manual_seed(777),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 2,
    "num_images_per_prompt": 1,
    "layers": 4,
    "resolution": 640,      # Using different bucket (640, 1024) to determine the resolution. For this version, 640 is recommended
    "cfg_normalize": True,  # Whether enable cfg normalization.
    "use_en_prompt": True,  # Automatic caption language if user does not provide caption
}

with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
print('spend', time.time() - start)

for i, image in enumerate(output_image):
    image.save(f"out{i}.png")
