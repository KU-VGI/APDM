import argparse
import os
import torch
import random
import csv

import numpy as np

from diffusers import StableDiffusionPipeline

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from diffusers import schedulers

import pandas as pd
from datetime import datetime
import json


class ImageDataset(Dataset):
    def __init__(self, path, map=None):
        self.path = path

        self.imgs = sorted(os.listdir(self.path))
        if map is not None:
            # map is json file
            with open(map, "r") as f:
                data = json.load(f)

            self.img_ids = []
            self.img_files = []
            self.captions = []
            self.full_paths = []


            for idx in range(len(data)):
                img_id = data[idx]["id"]
                img_file = data[idx]["id"] + ".jpg"
                caption = data[idx]["caption"]
                full_path = os.path.join(path, img_file)

                self.img_ids.append(img_id)
                self.img_files.append(img_file)
                self.captions.append(caption)
                self.full_paths.append(full_path)

    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        return self.img_ids[idx], self.img_files[idx], self.captions[idx], self.full_paths[idx]



def generate_samples(exp):
    '''
    Generate images using the given model and save them to the output directory
    
    Args:
        --model_path: path to the model (e.g. models/stable-diffusion-v1-5)
        --args: arguments for the generation
    args:
        --coco_map: path to the csv file containing the image and caption pairs
        --data_dir: path to the MS-COCO dataset directory
        --gen_batch_size: batch size for generation
        --output_dir: path to the output directory to save the generated images
        --resolution: resolution of the generated images
        --num_inference_steps: number of inference steps
        --guidance_scale: guidance scale for the diffusion model
    '''
    # setting on the main function
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # make directory for generated images
    dir_path = f"/data/slcks1/diffusion/tifa/results/{exp}"
    os.makedirs(dir_path, exist_ok=True)
    text_data = ImageDataset(dir_path, map="/data/slcks1/diffusion/tifa/tifa_v1.0/tifa_v1.0_text_inputs.json")
    loader = DataLoader(text_data, batch_size=3, shuffle=False, num_workers=4, drop_last=False)

    # Load the diffusion model
    pipe = StableDiffusionPipeline.from_pretrained("models/stable-diffusion-v1-5").to("cuda")
    pipe.set_progress_bar_config(disable=True)
    pipe.unet.load_state_dict(torch.load(f"experiments/{exp}/unet_799.pt"))
    pipe.safety_checker = None

    
    print("====================================")
    print("===========Start sampling===========")
    print("====================================")

    json_data = {}

    with torch.no_grad():
        for img_ids, img_files, captions, full_paths in tqdm(loader):
            images = pipe(list(captions), height=512, width=512,
                          num_inference_steps=20, guidance_scale=7.5).images

            for i, img in enumerate(images):
                img.save(full_paths[i])
                json_data[img_ids[i]] = img_files[i]

                del img
            del images

    del pipe

    with open(os.path.join(dir_path, f"{exp}_images.json"), "w") as f:
        json.dump(json_data, f, indent=4)

    print("====================================")
    print("==========Finish sampling===========")
    print("====================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate diffusion model")
    parser.add_argument("--exp", type=str, required=True, help="unique token (e.g., 'sks')")

    args = parser.parse_args()
    generate_samples(args.exp)