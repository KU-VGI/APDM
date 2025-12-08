# inital setting 
exp=$1           # e.g. apdm001
iter=799

args=`python gen_arguments.py $exp --mode protect`

echo $args

python protect.py --exp $exp $args

dreambooth_args=`python gen_arguments.py $exp --mode dreambooth`

echo $dreambooth_args

python train_dreambooth.py $dreambooth_args \
                            --output_dir "outputs/${exp}_${iter}it" \
                            --additional_unet_path "experiments/${exp}/unet_${iter}.pt"

eval_args=`python gen_arguments.py $exp --mode evaluate`

python evaluate_db.py --checkpoint outputs/${exp}_${iter}it \
                      $eval_args \
                      --output_dir fake_images/${exp}_${iter}_db \
                      --scheduler pndm \
                      --num_inference_steps 20 \
                      --seed 0 \
                      --dino_score \
                      --clip_score \
                      --exp ${exp} \
                      --brisque

python evaluate.py --coco_map coco_val2014_5k.csv \
                   --coco_fid_feat coco_5k_fid_feat.npy \
                   --checkpoint models/stable-diffusion-v1-5 \
                   --output_dir fake_images/${exp}_${iter} \
                   --seed 0 \
                   --fid \
                   --clip_score \
                   --additional_unet_path "experiments/${exp}/unet_${iter}.pt" \
                   --num_inference_steps 20 