import os

import torch
from torchvision import transforms
from torch.nn import functional as F

from transformers import ViTModel
from PIL import Image


def pil_to_tensor(path, transform):
    '''
    Convert PIL image to tensor
    
    Args:
        --path: path to the image
    '''
    imgs = sorted(os.listdir(path))
    images = [transform(Image.open(os.path.join(path, img)).convert("RGB")) for img in imgs]

    return torch.stack(images)


torch.no_grad()
def calculate_dino(ori_path, gen_path, args):
    '''
    Calculate the DINO score for the personalized model (from DreamBooth)
    
    Args:
        --args: arguments for the generation
    '''
    # setting on the main function
    # if args.seed is not None:
    #     torch.manual_seed(args.seed)
    #     torch.cuda.manual_seed(args.seed)
    #     random.seed(args.seed)
    #     np.random.seed(args.seed)

    # DINO Transforms
    T = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Load images
    ori_imgs = pil_to_tensor(ori_path, T).to(args.device)
    gen_imgs = pil_to_tensor(gen_path, T).to(args.device)

    # Load DINO ViT-S/16
    model = ViTModel.from_pretrained('facebook/dino-vits16').to(args.device)

    # Get DINO features
    with torch.no_grad():
        ori_feats = model(ori_imgs).last_hidden_state[:, 0, :] # batch_size x 768
        gen_feats = model(gen_imgs).last_hidden_state[:100, 0, :] # batch_size x 768

    # Calculate DINO score
    # calculate cosine similarity between the all pairs of (ori, gen) features
    cos_sims = F.cosine_similarity(ori_feats.unsqueeze(1), gen_feats.unsqueeze(0), dim=-1)
    cos_sims = cos_sims.mean()
    dino_score = cos_sims.detach().cpu().numpy()

    # print(f'Dino Score: {dino_score.item()}')

    return dino_score
    