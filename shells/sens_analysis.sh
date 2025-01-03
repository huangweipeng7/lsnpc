#!/usr/bin/env bash

# voc 2012
dataset='voc2007'


# for noise_type in 'pairflip'
# do
#     for img_encoder in 'resnet50'
#     do
#         for noise in 0.4
#         do
#         python3 -m training.train_hlc \
#         --dataset $dataset \
#         --noise_rate $noise \
#         --noise_type $noise_type \
#         --img_encoder $img_encoder \
#         --clf_name hlc \
#         --num_workers 4 \
#         --result_dir 'trained_models/hlc/' \
#         --delta 0.45 \
#         --epoch_update_start 5 \
#         --noisy_val true \
#         --n_train_epoch 20 \
#         --n_repeats 1 \
#         --eval_test_at_final_loop_only true \
#         --noisy_val false \
#         --seed 100
#         done
#     done
# done


# for noise_type in 'pairflip'
# do 
# 	for pretrained in 'hlc'
# 	do
# 		for nu0 in '1' '2.01' '3' '4'
# 		do
# 			for nu in '2.01' # '3' '4'
# 			do

# 			# if [ "$nu0" == "2.01" ] && [ "$nu" == "2.01" ]
# 			# then 
# 			# 	continue 1
# 			# fi
# 			pretrained_clf=$(ls ./trained_models/${pretrained}/${dataset}_${noise_type}_0.4_resnet50_ep20_rd0/*.pth)
# 			echo "Using pretrained classifier ${pretrained_clf}"

# 			python3 -m training.train_mlnlc \
# 			--batch_size 32 \
# 			--clf_name $pretrained \
# 			--dataset $dataset \
# 			--lr 5e-5 \
# 			--noise_rate 0.4 \
# 			--noise_type $noise_type \
# 			--num_workers 4 \
# 			--n_train_epoch 20 \
# 			--beta 0.01 \
# 			--omega 0.9 \
# 			--nu0 $nu0 \
# 			--nu $nu \
# 			--noisy_val false \
# 			--weight_decay 1e-5 \
# 			--img_encoder 'resnet50' \
# 			--semi_sup false \
# 			--result_dir './trained_models/mlnlc/' \
# 			--pretrained_clf ${pretrained_clf} \
# 			--n_repeats 5 \
# 			--grad_norm 2 \
# 			--eval_test_at_final_loop_only true \
# 			--seed 100
# 			done
# 		done
# 	done 
# done


for noise_type in 'symmetric'
do
   for pretrained in 'mlclf'
   do
       for nu0 in 1 2.01 3 4
       do
           for nu in 2.01 # 3 4
           do
           pretrained_clf=$(ls ./trained_models/${pretrained}/tomato_${noise_type}_0.3_levit_ep40_rd0/*.pth)
           echo "Using pretrained classifier ${pretrained_clf}"

           python3 -m training.train_mlnlc \
           --batch_size 32 \
           --clf_name $pretrained \
           --dataset tomato \
           --lr 5e-5 \
           --noise_rate 0.4 \
           --noise_type $noise_type \
           --num_workers 4 \
           --n_train_epoch 30 \
           --beta 0.01 \
           --omega 0.9 \
           --nu0 $nu0 \
           --nu $nu \
           --noisy_val false \
           --weight_decay 1e-5 \
           --img_encoder 'levit' \
           --semi_sup false \
           --result_dir './trained_models/mlnlc/' \
           --pretrained_clf ${pretrained_clf} \
           --n_repeats 5 \
           --grad_norm 2 \
           --eval_test_at_final_loop_only true \
           --seed 100
           done
       done
   done
done
