import os

import torch
from tqdm import tqdm

from diffusers import StableDiffusionPipeline
from diffusers import schedulers

from torchvision import transforms
from torch.nn import functional as F

from transformers import ViTModel
import random

import pandas as pd

from evaluation.dino import pil_to_tensor


def load_prompt_list(class_token, unique_token, diverse=False):
    '''
    Generate prompt list for the given class noun
    
    Args:
        --class_token: class noun (e.g., 'dog', 'cat', 'person')
        --unique_token: unique token (e.g., 'sks')
    '''

    object_class_list = ['backpack', 'bowl', 'clock', 'sneaker', 'candle', 'can', 'vase', 'teapot', 'toy']
    live_subject_class_list = ['cat']
    person_subject_class_list = ['person', 'dog', 'sneaker','cat', 'glasses', 'man', 'woman']

    if diverse:
        prompt_list = [
        'a {0} {1} in the jungle'.format(unique_token, class_token),
        'a {0} {1} in the snow'.format(unique_token, class_token),
        'a {0} {1} on the beach'.format(unique_token, class_token),
        'a {0} {1} on a cobblestone street'.format(unique_token, class_token),
        'a {0} {1} on top of pink fabric'.format(unique_token, class_token),
        'a {0} {1} on top of a wooden floor'.format(unique_token, class_token),
        'a {0} {1} with a city in the background'.format(unique_token, class_token),
        'a {0} {1} with a mountain in the background'.format(unique_token, class_token),
        'a {0} {1} with a blue house in the background'.format(unique_token, class_token),
        'a {0} {1} on top of a purple rug in a forest'.format(unique_token, class_token),
        'a {0} {1} wearing a red hat'.format(unique_token, class_token),
        'a {0} {1} wearing a santa hat'.format(unique_token, class_token),
        'a {0} {1} wearing a rainbow scarf'.format(unique_token, class_token),
        'a {0} {1} wearing a black top hat and a monocle'.format(unique_token, class_token),
        'a {0} {1} in a chef outfit'.format(unique_token, class_token),
        'a {0} {1} in a firefighter outfit'.format(unique_token, class_token),
        'a {0} {1} in a police outfit'.format(unique_token, class_token),
        'a {0} {1} wearing pink glasses'.format(unique_token, class_token),
        'a {0} {1} wearing a yellow shirt'.format(unique_token, class_token),
        'a {0} {1} in a purple wizard outfit'.format(unique_token, class_token),
        'a red {0} {1}'.format(unique_token, class_token),
        'a purple {0} {1}'.format(unique_token, class_token),
        'a shiny {0} {1}'.format(unique_token, class_token),
        'a wet {0} {1}'.format(unique_token, class_token),
        'a cube shaped {0} {1}'.format(unique_token, class_token)
        ]

        return prompt_list

    if class_token in object_class_list:
        # Object Prompts
        prompt_list = [
        'a {0} {1} in the jungle'.format(unique_token, class_token),
        'a {0} {1} in the snow'.format(unique_token, class_token),
        'a {0} {1} on the beach'.format(unique_token, class_token),
        'a {0} {1} on a cobblestone street'.format(unique_token, class_token),
        'a {0} {1} on top of pink fabric'.format(unique_token, class_token),
        'a {0} {1} on top of a wooden floor'.format(unique_token, class_token),
        'a {0} {1} with a city in the background'.format(unique_token, class_token),
        'a {0} {1} with a mountain in the background'.format(unique_token, class_token),
        'a {0} {1} with a blue house in the background'.format(unique_token, class_token),
        'a {0} {1} on top of a purple rug in a forest'.format(unique_token, class_token),
        'a {0} {1} with a wheat field in the background'.format(unique_token, class_token),
        'a {0} {1} with a tree and autumn leaves in the background'.format(unique_token, class_token),
        'a {0} {1} with the Eiffel Tower in the background'.format(unique_token, class_token),
        'a {0} {1} floating on top of water'.format(unique_token, class_token),
        'a {0} {1} floating in an ocean of milk'.format(unique_token, class_token),
        'a {0} {1} on top of green grass with sunflowers around it'.format(unique_token, class_token),
        'a {0} {1} on top of a mirror'.format(unique_token, class_token),
        'a {0} {1} on top of the sidewalk in a crowded street'.format(unique_token, class_token),
        'a {0} {1} on top of a dirt road'.format(unique_token, class_token),
        'a {0} {1} on top of a white rug'.format(unique_token, class_token),
        'a red {0} {1}'.format(unique_token, class_token),
        'a purple {0} {1}'.format(unique_token, class_token),
        'a shiny {0} {1}'.format(unique_token, class_token),
        'a wet {0} {1}'.format(unique_token, class_token),
        'a cube shaped {0} {1}'.format(unique_token, class_token)
        ]

    elif class_token in live_subject_class_list:
        # Live Subject Prompts
        prompt_list = [
        'a {0} {1} in the jungle'.format(unique_token, class_token),
        'a {0} {1} in the snow'.format(unique_token, class_token),
        'a {0} {1} on the beach'.format(unique_token, class_token),
        'a {0} {1} on a cobblestone street'.format(unique_token, class_token),
        'a {0} {1} on top of pink fabric'.format(unique_token, class_token),
        'a {0} {1} on top of a wooden floor'.format(unique_token, class_token),
        'a {0} {1} with a city in the background'.format(unique_token, class_token),
        'a {0} {1} with a mountain in the background'.format(unique_token, class_token),
        'a {0} {1} with a blue house in the background'.format(unique_token, class_token),
        'a {0} {1} on top of a purple rug in a forest'.format(unique_token, class_token),
        'a {0} {1} wearing a red hat'.format(unique_token, class_token),
        'a {0} {1} wearing a santa hat'.format(unique_token, class_token),
        'a {0} {1} wearing a rainbow scarf'.format(unique_token, class_token),
        'a {0} {1} wearing a black top hat and a monocle'.format(unique_token, class_token),
        'a {0} {1} in a chef outfit'.format(unique_token, class_token),
        'a {0} {1} in a firefighter outfit'.format(unique_token, class_token),
        'a {0} {1} in a police outfit'.format(unique_token, class_token),
        'a {0} {1} wearing pink glasses'.format(unique_token, class_token),
        'a {0} {1} wearing a yellow shirt'.format(unique_token, class_token),
        'a {0} {1} in a purple wizard outfit'.format(unique_token, class_token),
        'a red {0} {1}'.format(unique_token, class_token),
        'a purple {0} {1}'.format(unique_token, class_token),
        'a shiny {0} {1}'.format(unique_token, class_token),
        'a wet {0} {1}'.format(unique_token, class_token),
        'a cube shaped {0} {1}'.format(unique_token, class_token)
        ]

    elif class_token in person_subject_class_list:
        # Person Prompts
        prompt_list = [
        'a photo of {0} {1}'.format(unique_token, class_token),
        'a photo of {0} {1}'.format(unique_token, class_token),
        'a photo of {0} {1}'.format(unique_token, class_token),
        'a photo of {0} {1}'.format(unique_token, class_token),
        'a photo of {0} {1}'.format(unique_token, class_token),
        'a photo of {0} {1}'.format(unique_token, class_token),
        'a photo of {0} {1}'.format(unique_token, class_token),
        'a photo of {0} {1}'.format(unique_token, class_token),
        'a photo of {0} {1}'.format(unique_token, class_token),
        'a photo of {0} {1}'.format(unique_token, class_token),
        'a photo of {0} {1}'.format(unique_token, class_token),
        'a photo of {0} {1}'.format(unique_token, class_token),
        'a photo of {0} {1}'.format(unique_token, class_token),
        'a photo of {0} {1}'.format(unique_token, class_token),
        'a photo of {0} {1}'.format(unique_token, class_token),
        'a portrait of {0} {1}'.format(unique_token, class_token),
        'a portrait of {0} {1}'.format(unique_token, class_token),
        'a portrait of {0} {1}'.format(unique_token, class_token),
        'a portrait of {0} {1}'.format(unique_token, class_token),
        'a portrait of {0} {1}'.format(unique_token, class_token),
        'a portrait of {0} {1}'.format(unique_token, class_token),
        'a portrait of {0} {1}'.format(unique_token, class_token),
        'a portrait of {0} {1}'.format(unique_token, class_token),
        'a portrait of {0} {1}'.format(unique_token, class_token),
        'a portrait of {0} {1}'.format(unique_token, class_token),
        'a portrait of {0} {1}'.format(unique_token, class_token),
        'a portrait of {0} {1}'.format(unique_token, class_token),
        'a portrait of {0} {1}'.format(unique_token, class_token),
        'a portrait of {0} {1}'.format(unique_token, class_token),
        'a portrait of {0} {1}'.format(unique_token, class_token),
        ]

    return prompt_list


torch.no_grad()
def personalized_samplers(model_path, args):
    '''
    Calculate the DINO score for the personalized model (from DreamBooth)
    
    Args:
        --model_path: path to the model
        --args: arguments for the generation
    args:
        --class_noun: class noun (e.g., 'dog', 'cat', 'person')
        --identifier: unique token (e.g., 'sks')
        --output_dir: path to the output directory to save the generated images
        --resolution: resolution of the generated images
        --num_inference_steps: number of inference steps
        --guidance_scale: guidance scale for the diffusion model
        --additional_unet_path: path to the additional unet model
        --scheduler: scheduler type for the diffusion model
    '''
    # setting on the main function
    # if args.seed is not None:
    #     torch.manual_seed(args.seed)
    #     torch.cuda.manual_seed(args.seed)
    #     random.seed(args.seed)
    #     np.random.seed(args.seed)

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

    if args.diverse_prompt:
        prompt_list = load_prompt_list(args.class_noun, args.identifier, diverse=True)
    else:
        prompt_list = load_prompt_list(args.class_noun, args.identifier)

    # Make directory for generated images
    os.makedirs(os.path.join(args.output_dir, 'imgs'), exist_ok=True)

    print("====================================")
    print("===========Start sampling===========")
    print("====================================")

    if args.diverse_prompt:
        
        # DINO Transforms
        T = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # Load DINO ViT-S/16
        model = ViTModel.from_pretrained('facebook/dino-vits16').to(args.device)

        # set ori image
        ori_imgs = pil_to_tensor(args.data_dir, T).to(args.device)
        with torch.no_grad():
            ori_feats = model(ori_imgs).last_hidden_state[:, 0, :].unsqueeze(1) # batch x 768

        print("Generating diverse prompts")
        idx = 0
        results = []
        with torch.no_grad():
            for idx in tqdm(range(len(prompt_list))):
                dino_score = 0.0
                caption = prompt_list[idx]

                while dino_score < 0.05:
                    seed = random.randint(0, 100000)
                    generator = torch.manual_seed(seed)

                    image = pipe(caption, height=args.resolution, width=args.resolution, generator=generator,
                                    num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale).images[0]
                    img = T(image).unsqueeze(0).to(args.device)

                    # Get DINO features
                    with torch.no_grad():
                        gen_feat = model(img).last_hidden_state[:, 0, :].unsqueeze(0) # batch x 768

                    # Calculate DINO score
                    dino_score = F.cosine_similarity(ori_feats, gen_feat, dim=-1).mean().detach().cpu().numpy().item()

                print(f"Caption: {caption}, DINO Score: {dino_score}, Seed: {seed}")
                # # Save the generated images
                # for i, img in enumerate(images):
                #     img.save(os.path.join(args.output_dir, 'imgs', f"{idx}.png"))
                #     results.append((f"{idx}.png", captions[i]))
                #     idx += 1
                
                image.save(os.path.join(args.output_dir, 'imgs', f"{idx}.png"))
                results.append((f"{idx}.png", caption, seed))
                idx += 1

                del image
                del img

        del pipe

        pf = pd.DataFrame(results, columns=['image', 'caption', 'seed'])
        pf.to_csv(os.path.join(args.output_dir, 'image_caption_pairs.csv'), index=False, header=False)
    else:
        idx = 0
        results = []
        with torch.no_grad():
            for idx in tqdm(range(0, len(prompt_list), 3)):
                if idx+3 > len(prompt_list):
                    captions = prompt_list[idx:]
                else:
                    captions = prompt_list[idx:idx+3]
                images = pipe(captions, height=args.resolution, width=args.resolution,
                                num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale).images
                
                # Save the generated images
                for i, img in enumerate(images):
                    img.save(os.path.join(args.output_dir, 'imgs', f"{idx}.png"))
                    results.append((f"{idx}.png", captions[i]))
                    idx += 1

                    del img

                del images

        del pipe

        pf = pd.DataFrame(results, columns=['image', 'caption'])
        pf.to_csv(os.path.join(args.output_dir, 'image_caption_pairs.csv'), index=False, header=False)

    print("====================================")
    print("===========Sampling finished========")
    print("====================================")
