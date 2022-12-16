# %%

import json
import requests
import io
import base64
from PIL import Image, PngImagePlugin
import pickle

#%%
import math
import os
import sys
import traceback

import numpy as np
from PIL import Image, ImageOps, ImageChops

from modules import devices
from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images
from modules.shared import opts, state
import modules.shared as shared
import modules.processing as processing
from modules.ui import plaintext_to_html
import modules.images as images
import modules.scripts

# import devices
# from processing import Processed, StableDiffusionProcessingImg2Img, process_images
# from shared import opts, state
# import shared as shared
# import processing as processing
# from ui import plaintext_to_html
# import images as images
# import scripts

# %%

url = "http://127.0.0.1:7861"

payload = {
    "prompt": "puppy dog",
    "steps": 5
}

# with open('../response.pickle', 'rb') as handle:
#     response = pickle.load(handle)

# # %%
# with open('../payload.pickle', 'rb') as handle:
#     payload = pickle.load(handle)
# payload = {'outpath_samples': 'outputs/img2img-images', 'outpath_grids': 'outputs/img2img-grids',
#  'prompt': 'asdfasdf', 'negative_prompt': '',
#   'styles': ['None', 'None'], 
#   'seed': -1.0, 'subseed': -1.0, 'subseed_strength': 0,
#    'seed_resize_from_h': 0, 'seed_resize_from_w': 0, 'seed_enable_extras': False, 'sampler_index': 0, 
#    'batch_size': 4, 'n_iter': 1, 'steps': 50, 'cfg_scale': 7, 'width': 512, 'height': 512, 'restore_faces': True, 
#    'tiling': False, 
#    'init_images': [<PIL.Image.Image image mode=RGB size=2400x3600 at 0x2EA7934DCF0>], 
#    'mask': <PIL.Image.Image image mode=L size=2400x3600 at 0x2EA7931E890>,
#     'mask_blur': 20, 'inpainting_fill': 1, 'resize_mode': 0, 'denoising_strength': 0.75, 'inpaint_full_res': True, 
#     'inpaint_full_res_padding': 100, 'inpainting_mask_invert': 0}
# %%
with open('../payload.json') as handle:
    payload = json.load(handle)

del payload['sampler_index']
payload
# %% Img2Img
response = requests.post(url=f'{url}/sdapi/v1/img2img', json=payload)

r = response.json()

# Saving for referece, we must convert this json to equivalent behaviour as StableDiffusionProcessingImg2Img class
with open('response_api.json', 'w') as handle:
    json.dump(r, handle)   

# %%
for i in r['images']:
    image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))

    png_payload = {
        "image": "data:image/png;base64," + i
    }
    response2 = requests.post(url=f'{url}/sdapi/v1/png-info', json=png_payload)

    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("parameters", response2.json().get("info"))
    image.save('output.png', pnginfo=pnginfo)

# %%
response = requests.get('http://127.0.0.1:7861/sdapi/v1/progress?skip_current_image=false')

# %%
with open('../response_api.json') as handle:
    response_api = json.load(handle)
processed_images = [Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0]))) for i in response_api['images']]
generation_info_js = response_api['info']
processed_info = json.loads(generation_info_js)['infotexts']

# %% Original
response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)

r = response.json()


for i in r['images']:
    image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))

    png_payload = {
        "image": "data:image/png;base64," + i
    }
    response2 = requests.post(url=f'{url}/sdapi/v1/png-info', json=png_payload)

    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("parameters", response2.json().get("info"))
    image.save('output.png', pnginfo=pnginfo)
# %%
import json

generation_info_js_str = '''
{"prompt": "asd", "all_prompts": ["asd", "asd", "asd", "asd"], 
"negative_prompt": "", 
"seed": 1395414724, 
"all_seeds": [1395414724, 1395414725, 1395414726, 1395414727],
 "subseed": 4046790087, 
 "all_subseeds": [4046790087, 4046790088, 4046790089, 4046790090],
  "subseed_strength": 0, "width": 512, "height": 512, "sampler_index": 0, 
  "sampler": "Euler a", "cfg_scale": 7, "steps": 50, "batch_size": 4,
 "restore_faces": true, "face_restoration_model": null, 
 "sd_model_hash": "7460a6fa", "seed_resize_from_w": 0, "seed_resize_from_h": 0, "denoising_strength": 0.75, 
"extra_generation_params": {"Mask blur": 20}, "index_of_first_image": 1, 
"infotexts": ["asd\nSteps: 50, Sampler: Euler a, CFG scale: 7, Seed: 1395414724, Size: 512x512, Model hash: 7460a6fa, Batch size: 4, Batch pos: 0, Denoising strength: 0.75, Mask blur: 20", "asd\nSteps: 50, Sampler: Euler a, CFG scale: 7, Seed: 1395414724, Size: 512x512, Model hash: 7460a6fa, Batch size: 4, Batch pos: 0, Denoising strength: 0.75, Mask blur: 20", "asd\nSteps: 50, Sampler: Euler a, CFG scale: 7, Seed: 1395414725, Size: 512x512, Model hash: 7460a6fa, Batch size: 4, Batch pos: 1, Denoising strength: 0.75, Mask blur: 20", "asd\nSteps: 50, Sampler: Euler a, CFG scale: 7, Seed: 1395414726, Size: 512x512, Model hash: 7460a6fa, Batch size: 4, Batch pos: 2, Denoising strength: 0.75, Mask blur: 20", "asd\nSteps: 50, Sampler: Euler a, CFG scale: 7, Seed: 1395414727, Size: 512x512, Model hash: 7460a6fa, Batch size: 4, Batch pos: 3, Denoising strength: 0.75, Mask blur: 20"]}
'''

json.loads(generation_info_js_str)
# %%
