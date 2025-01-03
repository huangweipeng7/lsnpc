#!/usr/bin/env bash


dataset='voc2012'

for noise_type in 'symmetric' # 'pairflip'
do
  for noise in '0.4' #'0.5' '0.0'
  do
    if [ "$noise" == "0.0" ] && [ "$noise_type" == "pairflip" ]
    then
      echo "continue"
      continue 1
    fi

    for img_encoder in 'resnet50' # 'levit'  
    do
    python3 -m training.train_mlclf \
    --dataset $dataset \
    --noise_rate $noise \
    --noise_type $noise_type \
    --img_encoder $img_encoder \
    --batch_size 128 \
    --clf_name mlclf \
    --num_workers 4 \
    --result_dir 'trained_models/mlclf/' \
    --n_train_epoch 20 \
    --n_repeats 1 \
    --eval_test_at_final_loop_only true \
    --noisy_val false \
    --seed 100
    done
	done 
done



# for noise_type in 'symmetric' 'pairflip'
# do
#   for noise in '0.3' '0.5' '0.0'
#   do
#     if [ "$noise" == "0.0" ] && [ "$noise_type" == "pairflip" ]
#     then
#       continue 1
#     fi

#     for img_encoder in 'levit' 'resnet50'  
#     do

#     pretrained='mlclf'
#     pretrained_clf=$(ls ./trained_models/${pretrained}/${dataset}_${noise_type}_${noise}_${img_encoder}_ep20_rd0/*.pth)
# 		echo "Using pretrained classifier ${pretrained_clf}"

#     python3 -m training.finetune \
#     --dataset $dataset \
#     --noise_rate $noise \
#     --noise_type $noise_type \
#     --img_encoder $img_encoder \
#     --lr 5e-5 \
#     --batch_size 128 \
#     --post_model 'finetune' \
#     --pretrained_clf $pretrained_clf \
#     --clf_name $pretrained \
#     --num_workers 4 \
#     --result_dir 'trained_models/mlclf_finetune/' \
#     --n_train_epoch 10 \
#     --n_repeats 5 \
#     --eval_test_at_final_loop_only true \
#     --noisy_val false \
#     --seed 100
#     done
# 	done 
# done
