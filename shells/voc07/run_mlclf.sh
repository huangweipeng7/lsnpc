#!/usr/bin/env bash
 
# for noise_type in 'symmetric' 
# do
# 	for img_encoder in 'levit' #'resnet50' 'vit224'  
# 	do
# 		for noise in '0.0' '0.4' '0.5' '0.3'
# 		do
# 		python3 -m training.train_mlclf \
# 		--dataset voc2007 \
# 		--noise_rate $noise \
# 		--noise_type $noise_type \
# 		--img_encoder $img_encoder \
# 		--batch_size 128 \
# 		--clf_name mlclf \
# 		--num_workers 4 \
# 		--result_dir 'trained_models/mlclf/' \
# 		--n_train_epoch 20 \
# 		--n_repeats 1 \
# 		--eval_test_at_final_loop_only true \
# 		--noisy_val false \
# 		--seed 100
# 		done
# 	done 
# done
 
for noise_type in 'pairflip' 
do 
	for img_encoder in 'levit' #'resnet50' 'vit224'  
	do
		for noise in '0.4' '0.5' '0.3'
		do
		python3 -m training.train_mlclf \
		--dataset voc2007 \
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