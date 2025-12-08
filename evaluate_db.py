import os
import argparse
import random
import numpy as np
import torch

from evaluation.personalized_sampler import personalized_samplers
from evaluation.clip_score import calculate_clip_score
from evaluation.dino import calculate_dino

import pandas as pd


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Evaluate personalized model")
    parser.add_argument("--checkpoint", type=str, required=True, help="path to the model checkpoint")
    parser.add_argument("--class_noun", type=str, required=True, help="class noun (e.g., 'dog', 'cat', 'person')")
    parser.add_argument("--identifier", type=str, required=True, help="unique token (e.g., 'sks')")
    parser.add_argument("--data_dir", type=str, required=True, help="path to the real images for personalization")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--gen_batch_size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, required=True, help="DB_fake_images/(exp): path to the output directory to save the generated images")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--dino_score", action="store_true", help="calculate DINO score")
    parser.add_argument("--clip_score", action="store_true", help="calculate CLIP score")
    parser.add_argument("--additional_unet_path", type=str, default=None)
    parser.add_argument("--scheduler", type=str, default="pndm")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--brisque", action="store_true", help="calculate BRISQUE score")
    parser.add_argument("--diverse_prompt", action="store_true", help="generate diverse prompts")
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

    # Generate images using the given model
    personalized_samplers(args.checkpoint, args)

    # Calculate DINO score
    if args.dino_score:
        dino_score = calculate_dino(args.data_dir, os.path.join(args.output_dir, 'imgs'), args)
        print(f"DINO score: {dino_score:.4f}")
    else:
        dino_score = None
    
    # Calculate CLIP score
    if args.clip_score:
        clip_score = calculate_clip_score(args)

        if args.scheduler == "dpm_solver":
            print(f"DPM CLIP score: {clip_score:.4f}")
        else:
            print(f"CLIP-T score: {clip_score:.4f}")
    else:
        clip_score = None

    if args.brisque:
        from brisque import BRISQUE
        from PIL import Image
        images = os.listdir(os.path.join(args.output_dir, 'imgs'))
        brisuqe_fn = BRISQUE(url=False)
        brisque_scores = []
        for image in images:
            if image.endswith(".png") or image.endswith(".jpg"):
                image = Image.open(os.path.join(args.output_dir, 'imgs', image)).convert("RGB")
                brisque_score = brisuqe_fn.score(image)
                brisque_scores.append(brisque_score)
        # print top 30 BRISQUE scores sorted large to small
        arr = np.array(brisque_scores)
        idx = np.argpartition(arr, -10)[-10:]
        mean = arr[idx].mean()
        print(f"Top 30 BRISQUE scores: {arr[idx]}")
        print(f"Mean of top 30 BRISQUE scores: {mean:.4f}")
        # print mean of all BRISQUE scores
        brisque_score = np.mean(brisque_scores)
        print(f"BRISQUE score: {brisque_score:.4f}")
    else:
        brisque_score = None

    # exp,FID,CLIP,DINO,CLIP-I,CLIP-T, BRISQUE
    if not os.path.exists("results.csv"):
        results = pd.DataFrame(columns=["exp", "FID", "CLIP", "DINO", "CLIP-T", "BRISQUE"])
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
        results = results._append({"exp": exp, "DINO": dino_score, "CLIP-T": clip_score, "BRISQUE": brisque_score}, ignore_index=True)
    else:
        results.loc[results["exp"] == exp, "DINO"] = dino_score
        results.loc[results["exp"] == exp, "CLIP-T"] = clip_score
        results.loc[results["exp"] == exp, "BRISQUE"] = brisque_score
    results = results.sort_values(by="exp")
    results.to_csv("results.csv", index=False)


if __name__ == "__main__":
    args = parse_args()
    main(args) # pylint: disable=no-value-for-parameter
