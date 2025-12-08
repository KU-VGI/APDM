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


class ImageDataset(Dataset):
    def __init__(self, path, transform=None, map=None):
        self.path = path
        self.transform = transform

        self.imgs = sorted(os.listdir(self.path))
        if map is not None:
            # map is csv file, first column is image id, second column is caption text
            # extract image id for using selected images
            image_caption_pairs = []

            with open(map, mode='r') as f:
                reader = csv.reader(f)
                for row in reader:
                    img_file, caption = row[0], row[1]
                    full_path = os.path.join(path, img_file)
                    if os.path.exists(full_path):
                        image_caption_pairs.append((img_file, caption))
                    else:
                        RuntimeError(f"Image {img_file} does not exist in the directory {path}")
            
            # sort by image id
            image_caption_pairs.sort(key=lambda x: x[0])

            self.imgs = [pair[0] for pair in image_caption_pairs]
            self.captions = [pair[1] for pair in image_caption_pairs]
        else:
            self.captions = None
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.path, self.imgs[idx])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.captions is not None:
            return img, self.captions[idx]
        else:
            return img, "Dummy caption"



def generate_samples(model_path, args):
    '''
    Generate images using the given model and save them to the output directory
    
    Args:
        --model_path: path to the model
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
    # if args.seed is not None:
    #     torch.manual_seed(args.seed)
    #     torch.cuda.manual_seed(args.seed)
    #     random.seed(args.seed)
    #     np.random.seed(args.seed)

    # For image generation, prepare the MS-COCO dataset
    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=InterpolationMode.LANCZOS),
        transforms.ToTensor()
    ])
    dataset = ImageDataset(os.path.join(args.data_dir, "coco/val2014"), map=args.coco_map, transform=transform)
    loader = DataLoader(dataset, batch_size=args.gen_batch_size, shuffle=False, num_workers=4, drop_last=False)

    # Load the diffusion model
    pipe = StableDiffusionPipeline.from_pretrained(model_path).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    if args.additional_unet_path is not None:
        pipe.unet.load_state_dict(torch.load(args.additional_unet_path))
    pipe.safety_checker = None

    if args.scheduler == "pndm":
        pass
    elif args.scheduler == "ddim":
        pipe.scheduler = schedulers.DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
    elif args.scheduler == "dpm_solver":
        pipe.scheduler = schedulers.DPMSolverMultistepScheduler.from_pretrained(model_path, subfolder="scheduler")

    # Make directory for generated images
    os.makedirs(os.path.join(args.output_dir, 'imgs'), exist_ok=True)

    print("====================================")
    print("===========Start sampling===========")
    print("====================================")

    idx = 0
    results = []
    with torch.no_grad():
        for _, captions in tqdm(loader):
            images = pipe(list(captions), height=args.resolution, width=args.resolution,
                          num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale).images

            for i, img in enumerate(images):
                img.save(os.path.join(args.output_dir, 'imgs', f"{idx}.png"))
                results.append((f"{idx}.png", captions[i])) # for clip score
                idx += 1

                del img

            del images

    del pipe

    df = pd.DataFrame(results, columns=["image_id", "caption"])
    df.to_csv(os.path.join(args.output_dir, "image_caption_pairs.csv"), index=False, header=False)

    print("====================================")
    print("==========Finish sampling===========")
    print("====================================")
