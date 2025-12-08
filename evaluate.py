import os
import argparse
import random
import numpy as np
import torch

from evaluation.sampler import generate_samples
from evaluation.fid import calculate_fid
from evaluation.clip_score import calculate_clip_score

import pandas as pd


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Evaluate diffusion model")
    parser.add_argument("--coco_map", type=str, default="coco_val2014_5k.csv", choices=['coco_val2014_5k.csv', 'coco_val2014_30k.csv', 'coco_val2014.csv'], help="csv file containing the image and caption pairs")
    parser.add_argument("--coco_fid_feat", type=str, default="coco_5k_fid_feat.npy", help="path to the pre-extracted features for the MS-COCO dataset")
    parser.add_argument("--checkpoint", type=str, required=True, help="path to the model checkpoint")
    parser.add_argument("--data_dir", type=str, default="/local_datasets", help="path to the MS-COCO dataset directory")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gen_batch_size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, required=True, help="fake_images/(exp): path to the output directory to save the generated images")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--fid", action="store_true", help="calculate FID score")
    parser.add_argument("--clip_score", action="store_true", help="calculate CLIP score")
    parser.add_argument("--additional_unet_path", type=str, default=None)
    parser.add_argument("--scheduler", type=str, default="pndm")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--exp", type=str, default=None, help="experiment name")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

def main(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    # set device
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # root of coco map files
    args.coco_map = os.path.join("coco_list", args.coco_map)

    # Generate images using the given model
    # os.makedirs(os.path.join(args.output_dir, 'imgs'), exist_ok=True)
    if os.path.exists(os.path.join(args.output_dir, 'imgs')):
        print("Output directory already exists. Skipping image generation.")
    else:
        generate_samples(args.checkpoint, args)

    # Calculate FID score
    if args.fid:
        fid_score = calculate_fid(args)
        print(f"FID score: {fid_score:.2f}")

    # Calculate CLIP score
    if args.clip_score:
        clip_score = calculate_clip_score(args)
        print(f"CLIP score: {clip_score:.4f}")

    # exp,FID,CLIP,DINO,CLIP-I,CLIP-T, BRISQUE
    if not os.path.exists("results.csv"):
        results = pd.DataFrame(columns=["exp", "FID", "CLIP", "DINO", "CLIP-I", "CLIP-T", "BRISQUE"])
    else:
        results = pd.read_csv("results.csv")
    
    if args.exp is not None:
        exp = args.exp
    else:
        if args.additional_unet_path is not None:
            exp = args.additional_unet_path.split("/")[-2]
        else:
            exp = args.checkpoint.split("/")[-1]

    print(exp)
    if exp not in results["exp"].values:
        results = results._append({"exp": exp, "FID": fid_score, "CLIP": clip_score}, ignore_index=True)
    else:
        results.loc[results["exp"] == exp, "FID"] = fid_score
        results.loc[results["exp"] == exp, "CLIP"] = clip_score
    results = results.sort_values(by="exp")
    results.to_csv("results.csv", index=False)



if __name__ == "__main__":
    args = parse_args()
    main(args) # pylint: disable=no-value-for-parameter
