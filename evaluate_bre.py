import os
import argparse
import random
import numpy as np
import torch

from evaluation.personalized_sampler import personalized_samplers
from evaluation.clip_score import calculate_clip_score
from evaluation.dino import calculate_dino



def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Evaluate personalized model")
    parser.add_argument("--checkpoint", type=str, help="path to the model checkpoint")
    parser.add_argument("--class_noun", default="person", type=str, help="class noun (e.g., 'dog', 'cat', 'person')")
    parser.add_argument("--identifier", default="sks", type=str, help="unique token (e.g., 'sks')")
    parser.add_argument("--data_dir", default="data/n000050/VGGFace2/n000057/set_A", type=str, help="path to the real images for personalization")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--gen_batch_size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, required=True, help="DB_fake_images/(exp): path to the output directory to save the generated images")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--dino_score", action="store_true", help="calculate DINO score")
    parser.add_argument("--clip_I_score", action="store_true", help="calculate CLIP-I score")
    parser.add_argument("--clip_score", action="store_true", help="calculate CLIP score")
    parser.add_argument("--additional_unet_path", type=str, default=None)
    parser.add_argument("--scheduler", type=str, default="pndm")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--brisque", action="store_true", help="calculate BRISQUE score")

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
    # personalized_samplers(args.checkpoint, args)

    # Calculate DINO score
    if args.dino_score:
        dino_score = calculate_dino(args.data_dir, os.path.join(args.output_dir, 'imgs'), args)

        if args.scheduler == "dpm_solver":
            print(f"DPM DINO score: {dino_score:.4f}")
        else:
            print(f"DINO score: {dino_score:.4f}")

    # Calculate CLIP score
    if args.clip_score:
        clip_score = calculate_clip_score(args)

        if args.scheduler == "dpm_solver":
            print(f"DPM CLIP score: {clip_score:.4f}")
        else:
            print(f"CLIP-T score: {clip_score:.4f}")

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
        brisque_score = np.mean(brisque_scores)
        print(f"BRISQUE score: {brisque_score:.4f}")


if __name__ == "__main__":
    args = parse_args()
    main(args) # pylint: disable=no-value-for-parameter
