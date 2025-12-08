import torch
import random
import os
import itertools
import sys

import numpy as np
import torch.nn.functional as F

from argparse import ArgumentParser
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from PIL import Image
from torchvision.utils import save_image
from torchvision import transforms
from datetime import datetime
from diffusers.schedulers import DDPMScheduler
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.Resize(512), 
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.images = os.listdir(data_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.data_dir, self.images[idx])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

class PairedImageDataset(Dataset):
    def __init__(self, data_dir1, data_dir2, size=512):
        self.data_dir1 = data_dir1
        self.data_dir2 = data_dir2
        self.transform = transforms.Compose([transforms.Resize(size),
                                             transforms.CenterCrop((size, size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.images1 = os.listdir(data_dir1)
        self.images2 = os.listdir(data_dir2)

        assert len(self.images1) == len(self.images2)
    
    def __len__(self):
        return len(self.images1)
    
    def __getitem__(self, idx):
        img1 = Image.open(os.path.join(self.data_dir1, self.images1[idx])).convert("RGB")
        img2 = Image.open(os.path.join(self.data_dir2, self.images2[idx])).convert("RGB")
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2

def encode_prompt(tokenizer, text_encoder, prompt, do_classifier_free_guidance, num_images_per_prompt=1,):
    text_inputs = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").to("cuda")

    text_input_ids = text_inputs.input_ids
    untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

    if untruncated_ids.shape[1] > text_input_ids.shape[1] and not torch.equal(text_input_ids, untruncated_ids):
        removed_text = tokenizer.batch_decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
        print(f"The following part of your input was truncated because CLIP can only handle sequences up to {tokenizer.model_max_length} tokens: {removed_text}")
    

    prompt_embeds = text_encoder(text_input_ids.to("cuda"), attention_mask=None)[0]

    bs_embeds, seq_len, _ = prompt_embeds.shape

    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embeds * num_images_per_prompt, seq_len, -1)

    if do_classifier_free_guidance:
        negative_text_inputs = tokenizer("", padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt").to("cuda")
        negative_prompt_embeds = text_encoder(negative_text_inputs.input_ids.to("cuda"), attention_mask=None)[0]
        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(bs_embeds * num_images_per_prompt, seq_len, -1)
    else:
        negative_prompt_embeds = None

    return prompt_embeds, negative_prompt_embeds

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="models/stable-diffusion-v1-5")
    parser.add_argument("--instance_data_dir", type=str, default="data/person")
    parser.add_argument("--instance_prompt", type=str, default="a photo of sks person")
    parser.add_argument("--with_prior_preservation", action="store_true")
    parser.add_argument("--prior_loss_weight", type=float, default=1.0) 
    parser.add_argument("--class_data_dir", type=str, default="class_images/person")
    parser.add_argument("--exp", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--iter", type=int, default=801)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_text_encoder", action="store_true")
    parser.add_argument("--num_inner_iter", type=int, default=30)
    parser.add_argument("--negative_loss", action="store_true")
    parser.add_argument("--print_freq", type=int, default=10)
    parser.add_argument("--save_freq", type=int, default=100000)
    parser.add_argument("--relu_bound", type=float, default=None)
    parser.add_argument("--class_prompt", type=str, default="a photo of a person")
    parser.add_argument("--grad_accum_type", type=str, default="sum")
    parser.add_argument("--unfreeze", nargs="+", default=None, help="List of layers to unfreeze")
    parser.add_argument("--in_ppl", action="store_true")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--loss_dpo", action="store_true")
    parser.add_argument("--loss_dpo_paired_dataset", action="store_true")
    parser.add_argument("--loss_dpo_paired_dataset_dir", type=str, default="paired_class_images/dog")
    parser.add_argument("--loss_dpo_beta", type=float, default=100)
    args = parser.parse_args()
    return args

def sample_data(loader):
    while True:
        for data in loader:
            yield data

def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.exists(args.class_data_dir):
        os.makedirs(args.class_data_dir)
    cur_class_images = len(list(os.listdir(args.class_data_dir)))

    if cur_class_images < args.num_samples:
        pipe = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, safety_checker=None).to("cuda")
        pipe.safty_checker = None
        pipe.set_progress_bar_config(disable=True)

        num_new_images = args.num_samples - cur_class_images
        print(f"Number of new images to generate: {num_new_images}")

        with torch.no_grad():
            for num in tqdm(range(num_new_images)):
                imgs = pipe(prompt=args.class_prompt, num_inference_steps=50).images[0]
                imgs.save(os.path.join(args.class_data_dir, f"{cur_class_images+num}.png"))

    pipe = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, safety_checker=None).to("cuda")
    pipe.safty_checker = None

    image_dataset = ImageDataset(args.instance_data_dir)
    image_loader = DataLoader(image_dataset, batch_size=args.batch_size, shuffle=True)
    image_loader = sample_data(image_loader)

    if args.loss_dpo and args.loss_dpo_paired_dataset:
        paired_image_dataset = PairedImageDataset(args.instance_data_dir, args.loss_dpo_paired_dataset_dir)
        paired_image_loader = DataLoader(paired_image_dataset, batch_size=args.batch_size, shuffle=True)
        paired_image_loader = sample_data(paired_image_loader)

    if args.with_prior_preservation:
        class_image_dataset = ImageDataset(args.class_data_dir)
        class_image_loader = DataLoader(class_image_dataset, batch_size=args.batch_size, shuffle=True)
        class_image_loader = sample_data(class_image_loader)
    
    if args.exp is None:
        save_path = f"experiments/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        save_path = f"experiments/{args.exp}"

    os.makedirs(save_path, exist_ok=True)
    with open(f"{save_path}/args.txt", "w") as f:
        f.write(str(args))
    
    ddpm_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    optimizer = torch.optim.AdamW(itertools.chain(pipe.unet.parameters(), pipe.text_encoder.parameters), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8) if args.train_text_encoder else torch.optim.AdamW(pipe.unet.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8)
    
    if args.loss_dpo:
        unet_source = deepcopy(pipe.unet)
        unet_source.requires_grad_(False)
        unet_source.eval()

    start_time = datetime.now()
    vae = pipe.vae
    for param in vae.parameters():
        param.requires_grad = False
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    for param in text_encoder.parameters():
        param.requires_grad = False
    for i in range(args.iter):
        optimizer.zero_grad()
        unet_temp = deepcopy(pipe.unet)
        optimizer_temp = torch.optim.AdamW(unet_temp.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8)
        for j in range(args.num_inner_iter):
            optimizer_temp.zero_grad()
            data = next(image_loader).to("cuda")

            t = torch.randint(0, ddpm_scheduler.config.num_train_timesteps, (args.batch_size,), device="cuda").long()
            z_0 = vae.encode(data).latent_dist.sample() * vae.config.scaling_factor
            eps = torch.randn_like(z_0)
            z_t = ddpm_scheduler.add_noise(z_0, eps, t)

            prompt_embeds, _ = encode_prompt(tokenizer, text_encoder, args.instance_prompt, False)

            model_pred = unet_temp(z_t, t, encoder_hidden_states=prompt_embeds, return_dict=False)[0]

            if ddpm_scheduler.config.prediction_type == "epsilon":
                target = eps
            elif ddpm_scheduler.config.prediction_type == "v_prediction":
                target = ddpm_scheduler.get_velocity(model_pred, eps, t)

            loss_db = F.mse_loss(model_pred, target, reduction="mean")
            if args.with_prior_preservation:
                class_data = next(class_image_loader).to("cuda")
                z_0_cls = vae.encode(class_data).latent_dist.sample() * vae.config.scaling_factor
                eps_cls = torch.randn_like(z_0_cls)
                z_t_cls = ddpm_scheduler.add_noise(z_0_cls, eps_cls, t)
                prompt_embeds_cls, _ = encode_prompt(tokenizer, text_encoder, args.class_prompt, False)
                model_pred_cls = unet_temp(z_t_cls, t, encoder_hidden_states=prompt_embeds_cls, return_dict=False)[0]
                if ddpm_scheduler.config.prediction_type == "epsilon":
                    target_cls = eps_cls
                elif ddpm_scheduler.config.prediction_type == "v_prediction":
                    target_cls = ddpm_scheduler.get_velocity(model_pred_cls, eps_cls, t)
                loss_cls = F.mse_loss(model_pred_cls, target_cls, reduction="mean")
                loss_db = loss_db + args.prior_loss_weight * loss_cls
            loss_db.backward()
            optimizer_temp.step()


            data = next(image_loader).to("cuda")
            t = torch.randint(0, ddpm_scheduler.config.num_train_timesteps, (args.batch_size,), device="cuda").long()
            z_0 = vae.encode(data).latent_dist.sample() * vae.config.scaling_factor
            eps = torch.randn_like(z_0)
            z_t = ddpm_scheduler.add_noise(z_0, eps, t)

            prompt_embeds, _ = encode_prompt(tokenizer, text_encoder, args.instance_prompt, False)

            model_pred = unet_temp(z_t, t, encoder_hidden_states=prompt_embeds, return_dict=False)[0]

            if ddpm_scheduler.config.prediction_type == "epsilon":
                target = eps
            elif ddpm_scheduler.config.prediction_type == "v_prediction":
                target = ddpm_scheduler.get_velocity(model_pred, eps, t)

            loss_neg = 0
            if args.negative_loss:
                if args.relu_bound is None or args.relu_bound == 0:
                    loss_neg = -F.mse_loss(model_pred, target, reduction="mean")
                else:
                    loss_neg = F.relu(args.relu_bound - F.mse_loss(model_pred, target, reduction="mean"))

            if args.in_ppl:
                class_data = next(class_image_loader).to("cuda")
                z_0_cls = vae.encode(class_data).latent_dist.sample() * vae.config.scaling_factor
                eps_cls = torch.randn_like(z_0_cls)
                z_t_cls = ddpm_scheduler.add_noise(z_0_cls, eps_cls, t)
                prompt_embeds_cls, _ = encode_prompt(tokenizer, text_encoder, args.class_prompt, False)
                model_pred_cls = unet_temp(z_t_cls, t, encoder_hidden_states=prompt_embeds_cls, return_dict=False)[0]
                if ddpm_scheduler.config.prediction_type == "epsilon":
                    target_cls = eps_cls
                elif ddpm_scheduler.config.prediction_type == "v_prediction":
                    target_cls = ddpm_scheduler.get_velocity(model_pred_cls, eps_cls, t)
                loss_cls = F.mse_loss(model_pred_cls, target_cls, reduction="mean")
                loss_neg = loss_neg + args.prior_loss_weight * loss_cls
            
            if args.loss_dpo:
                if args.loss_dpo_paired_dataset:
                    unsafe_data, safe_data = next(paired_image_loader)
                    unsafe_data = unsafe_data.to("cuda")
                    safe_data = safe_data.to("cuda")
                else:
                    unsafe_data = next(image_loader).to("cuda")
                    safe_data = next(class_image_loader).to("cuda")
                with torch.no_grad():
                    z_0_unsafe = vae.encode(unsafe_data).latent_dist.sample() * vae.config.scaling_factor
                    z_0_safe = vae.encode(safe_data).latent_dist.sample() * vae.config.scaling_factor
                eps = torch.randn_like(z_0_unsafe)
                t = torch.randint(0, ddpm_scheduler.config.num_train_timesteps, (args.batch_size,), device="cuda").long()
                z_t_unsafe = ddpm_scheduler.add_noise(z_0_unsafe, eps, t)
                z_t_safe = ddpm_scheduler.add_noise(z_0_safe, eps, t)
                
                """
                in duo,
                model_pred: current_model
                refer_pred: pre-trained model

                pred: negative (unsafe)
                base: positive (safe)

                loss_base = loss_model_base - loss_refer_base (current_safe - pre-trained_safe)
                loss_pred = loss_model_pred - loss_refer_pred (current_unsafe - pre-trained_unsafe)
                diff = loss_base - loss_pred
                """

                with torch.no_grad():
                    model_pred_unsafe_source = unet_source(z_t_unsafe, t, encoder_hidden_states=prompt_embeds, return_dict=False)[0]
                    model_pred_safe_source = unet_source(z_t_safe, t, encoder_hidden_states=prompt_embeds, return_dict=False)[0]

                model_pred_unsafe_target = unet_temp(z_t_unsafe, t, encoder_hidden_states=prompt_embeds, return_dict=False)[0]
                model_pred_safe_target = unet_temp(z_t_safe, t, encoder_hidden_states=prompt_embeds, return_dict=False)[0]
                loss_dpo = F.mse_loss(eps, model_pred_safe_target, reduction="none") - F.mse_loss(eps, model_pred_safe_source, reduction="none") \
                        - F.mse_loss(eps, model_pred_unsafe_target, reduction="none") + F.mse_loss(eps, model_pred_unsafe_source, reduction="none")
                loss_dpo = -1 * F.logsigmoid(-1 * args.loss_dpo_beta * loss_dpo)
                loss_dpo = loss_dpo.mean()
                loss_neg = loss_neg + loss_dpo

            loss_neg.backward()

            with torch.no_grad():
                for (name1, param1), (_, param2) in zip(pipe.unet.named_parameters(), unet_temp.named_parameters()):
                    if param2.grad is not None:
                        if args.unfreeze is None:
                            if param1.grad is None:
                                param1.grad = param2.grad
                            else:
                                param1.grad += param2.grad
                        elif any(name in name1 for name in args.unfreeze):
                            if param1.grad is None:
                                param1.grad = param2.grad
                            else:
                                param1.grad += param2.grad

                if args.train_text_encoder:
                    for (param1, param2) in zip(pipe.text_encoder.parameters(), text_encoder.parameters()):
                        if param2.grad is not None:
                            if param1.grad is None:
                                param1.grad = param2.grad
                            else:
                                param1.grad += param2.grad

            if (i * args.num_inner_iter + j) % args.print_freq == 0:
                print(f"Iter {str(i).zfill(3)}/{str(j).zfill(3)}: Loss (DB): {loss_db.item():.6f}, Loss (NEG): {loss_neg.item():.6f}, Time: {(datetime.now() - start_time) / (i * args.num_inner_iter + j + 1)}/it, ETA: {(datetime.now() - start_time) / (i * args.num_inner_iter + j + 1) * (args.iter * args.num_inner_iter - i * args.num_inner_iter - j)}") 
                sys.stdout.flush()

        if args.grad_accum_type == "mean":
            with torch.no_grad():
                for param in pipe.unet.parameters():
                    param.grad /= args.num_inner_iter

                if args.train_text_encoder:
                    for param in pipe.text_encoder.parameters():
                        param.grad /= args.num_inner_iter
        optimizer.step()

        if (i != 0 and i % args.save_freq == 0) or i == args.iter - 1:
            torch.save(pipe.unet.state_dict(), f"{save_path}/unet_{str(i).zfill(3)}.pt")
            if args.train_text_encoder:
                torch.save(pipe.text_encoder.state_dict(), f"{save_path}/text_encoder_{str(i).zfill(3)}.pt")


if __name__ == "__main__":
    args = parse_args()
    main(args)