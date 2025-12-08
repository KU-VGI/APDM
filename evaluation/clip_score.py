import torch
import clip
import os

from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from evaluation.sampler import ImageDataset


def tensor2pil(image: torch.Tensor):
    ''' output image : tensor to PIL
    '''
    if isinstance(image, list) or image.ndim == 4:
        return [tensor2pil(im) for im in image]

    assert image.ndim == 3
    output_image = Image.fromarray(((image + 1.0) * 127.5).clamp(
        0.0, 255.0).to(torch.uint8).permute(1, 2, 0).detach().cpu().numpy())
    return output_image

    

@torch.no_grad()
def calculate_clip_score(args):
    '''
    Calculate the CLIP score for the generated images (CLIP-T)

    Args:
        --args: arguments for the generation
    args:
        --output_dir: path to the generated images directory
        --batch_size: batch size for feature extraction        
    '''
    # setting on the main function
    # if args.seed is not None:
    #     torch.manual_seed(args.seed)
    #     torch.cuda.manual_seed(args.seed)
    #     random.seed(args.seed)
    #     np.random.seed(args.seed)
    
    # Load CLIP model
    model, preprocess = clip.load("ViT-L/14", device=args.device)

    # Load generated data for clip score
    dataset = ImageDataset(path=os.path.join(args.output_dir, 'imgs'), map=os.path.join(args.output_dir, "image_caption_pairs.csv"), transform=preprocess)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    # Calculate the CLIP score
    cos_sims = []
    count = 0

    with torch.no_grad():
        for imgs, captions in tqdm(loader):
            imgs = imgs.to(args.device)
            # captions = [t[0] for t in captions]
            text_tokens = clip.tokenize(captions, truncate=True).to(args.device)

            img_feat = model.encode_image(imgs)
            text_feat = model.encode_text(text_tokens)

            similarity = F.cosine_similarity(img_feat, text_feat)
            cos_sims.append(similarity)
            count += similarity.shape[0]
        
    cos_sims = torch.cat(cos_sims, dim=0).mean()
    clip_score = cos_sims.detach().cpu().numpy()

    # print(f"CLIP-T: {clip_score:.4f}")

    return clip_score
