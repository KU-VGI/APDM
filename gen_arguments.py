import argparse

def check_exp(exp, exp_list, prefix="apdm", zfill=3):
    return exp in [f"{prefix}{str(e).zfill(zfill)}" for e in exp_list]

def get_arguments(exp):
    default_args = {
        "batch_size": 1,
        "iter": 800,
        "num_inner_iter": 30,
        "with_prior_preservation": "",
        "prior_loss_weight": 1.0,
        "grad_accum_type": "sum",
        "in_ppl": "",
    }

    args = default_args

    ### Learning 
    if check_exp(exp, []): # for sd v2.1
        args["num_inner_iter"] = 10
    elif check_exp(exp, [1, 2, 3, 4, 5, 6, 7, 8]):
        args["num_inner_iter"] = 20
    

    ### dpo
    if check_exp(exp, [1, 2, 3, 4, 5, 6, 7, 8,]):
        args["loss_dpo"] = ""

    if check_exp(exp, [1, 2, 3, 4, 5, 6, 7, 8,]):
        args["loss_dpo_beta"] = 1
    
    if check_exp(exp, [1, 2, 3, 4, 5, 6, 7, 8,]):
        args["loss_dpo_paired_dataset"] = ""
    
    
    if check_exp(exp, [1]):
        args["loss_dpo_paired_dataset_dir"] = f"paired_set/person"
        args["instance_data_dir"] = "data/person/set_A"
    elif check_exp(exp, [2]):
        args["loss_dpo_paired_dataset_dir"] = f"paired_set/person2"
        args["instance_data_dir"] = "data/person2/set_A"
    elif check_exp(exp, [3]):
        args["loss_dpo_paired_dataset_dir"] = f"paired_set/person3"
        args["instance_data_dir"] = "data/person3/set_A"
    elif check_exp(exp, [4]):
        args["loss_dpo_paired_dataset_dir"] = f"paired_set/person4"
        args["instance_data_dir"] = "data/person4/set_A"
    elif check_exp(exp, [5]):
        args["loss_dpo_paired_dataset_dir"] = f"paired_set/dog"
        args["instance_data_dir"] = "data/dog"
    elif check_exp(exp, [6]):
        args["loss_dpo_paired_dataset_dir"] = f"paired_set/dog2"
        args["instance_data_dir"] = "data/dog2"
    elif check_exp(exp, [7]):
        args["loss_dpo_paired_dataset_dir"] = f"paired_set/dog3"
        args["instance_data_dir"] = "data/dog3"
    elif check_exp(exp, [8]):
        args["loss_dpo_paired_dataset_dir"] = f"paired_set/dog4"
        args["instance_data_dir"] = "data/dog4"


    ### prompt
    if check_exp(exp, [1, 2, 3, 4,]):
        class_word = "person"
        special_word = "sks"

        args["instance_prompt"] = f"\"a photo of {special_word} {class_word}\"".replace(" ", "_")
        args["class_prompt"] = f"\"a photo of {class_word}\"".replace(" ", "_")
        args["class_data_dir"] = "class_images/person"

    elif check_exp(exp, [5, 6, 7, 8,]):
        class_word = "dog"
        special_word = "sks"

        args["instance_prompt"] = f"\"a photo of {special_word} {class_word}\"".replace(" ", "_")
        args["class_prompt"] = f"\"a photo of {class_word}\"".replace(" ", "_")
        args["class_data_dir"] = "class_images/dog"

    if check_exp(exp, []):
        args["pretrained_model_name_or_path"] = "models/stable-diffusion-2-1-base"

    return " ".join([f"--{k} {v}" for k, v in args.items()])
    
def get_arguments_eval(exp):    
    default_args = {
        "identifier": "sks",
    }
    
    args = default_args

    if check_exp(exp, [1]):
        args["data_dir"] = "data/person/set_A"
        args["class_noun"] = "person"
    elif check_exp(exp, [2]):
        args["data_dir"] = "data/person2/set_A"
        args["class_noun"] = "person"
    elif check_exp(exp, [3]):
        args["data_dir"] = "data/person3/set_A"
        args["class_noun"] = "person"
    elif check_exp(exp, [4]):
        args["data_dir"] = "data/person4/set_A"
        args["class_noun"] = "person"
    elif check_exp(exp, [5]):
        args["data_dir"] = "data/dog"
        args["class_noun"] = "dog"
    elif check_exp(exp, [6]):
        args["data_dir"] = "data/dog2"
        args["class_noun"] = "dog"
    elif check_exp(exp, [7]):
        args["data_dir"] = "data/dog3"
        args["class_noun"] = "dog"
    elif check_exp(exp, [8]):
        args["data_dir"] = "data/dog4"
        args["class_noun"] = "dog"

    return " ".join([f"--{k} {v}" for k, v in args.items()])


def get_arguments_db(exp):
    default_args = {
        "pretrained_model_name_or_path": "models/stable-diffusion-v1-5",
        "resolution": 512,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 5e-6,
        "lr_scheduler": "constant",
        "lr_warmup_steps": 0,
        "max_train_steps": 800,
        "with_prior_preservation": "",
        "prior_loss_weight": 1.0,
        "num_class_images": 200,
        "seed": 0,
    }

    args = default_args

    if check_exp(exp, [1]):
        args["instance_data_dir"] = "data/person/set_A"
    elif check_exp(exp, [2]):
        args["instance_data_dir"] = "data/person2/set_A"
    elif check_exp(exp, [3]):
        args["instance_data_dir"] = "data/person3/set_A"
    elif check_exp(exp, [4]):
        args["instance_data_dir"] = "data/person4/set_A"
    elif check_exp(exp, [5]):
        args["instance_data_dir"] = "data/dog"
    elif check_exp(exp, [6]):
        args["instance_data_dir"] = "data/dog2"
    elif check_exp(exp, [7]):
        args["instance_data_dir"] = "data/dog3"
    elif check_exp(exp, [8]):
        args["instance_data_dir"] = "data/dog4"


    if check_exp(exp, [1, 2, 3, 4]):
        class_word = "person"
        special_word = "sks"

        args["class_data_dir"] = "class_images/person"
    elif check_exp(exp, [5, 6, 7, 8]):
        class_word = "dog"
        special_word = "sks"

        args["class_data_dir"] = "class_images/dog"


    args["instance_prompt"] = f"\"a photo of {special_word} {class_word}\"".replace(" ", "_")
    args["class_prompt"] = f"\"a photo of {class_word}\"".replace(" ", "_")

    if check_exp(exp, []):
        args["pretrained_model_name_or_path"] = "models/stable-diffusion-2-1-base"


    return " ".join([f"--{k} {v}" for k, v in args.items()])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp", type=str)
    parser.add_argument("--mode", type=str, choices=["protect", "dreambooth", "evaluate"], default="protect")
    args = parser.parse_args()

    if args.mode == "protect":
        print(get_arguments(args.exp))
    elif args.mode == "dreambooth":
        print(get_arguments_db(args.exp))
    elif args.mode == "evaluate":
        print(get_arguments_eval(args.exp))