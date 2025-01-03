#!/usr/bin/env bash

# For voc2007
#for noise_type in 'pairflip' # 'symmetric' 
#do
#	for img_encoder in 'resnet50' 'vit224'  
#	do
#		for noise in '0.0' # '0.4' '0.5' '0.3'
#		do
#		python3 -m training.train_mlclf \
#		--dataset voc2007 \
#		--noise_rate $noise \
#		--noise_type $noise_type \
#		--img_encoder $img_encoder \
#		--batch_size 128 \
#		--clf_name mlclf \
#		--num_workers 4 \
#		--result_dir 'trained_models/mlclf/' \
#		--n_train_epoch 20 \
#		--n_repeats 1 \
#		--eval_test_at_final_loop_only true \
#		--noisy_val false \
#		--seed 100
#		done
#	done 
#done

###############
# For tomato
#for noise_type in 'pairflip' 'symmetric' 
#do
#	for img_encoder in 'resnet50' 'vit224'  
#	do
#		for noise in '0.0' '0.4' '0.5' '0.3'
#		do
#		python3 -m training.train_mlclf \
#		--dataset tomato \
#		--lr 1e-5 \
#		--noise_rate $noise \
#		--noise_type $noise_type \
#		--img_encoder $img_encoder \
#		--batch_size 32 \
#		--clf_name mlclf \
#		--num_workers 4 \
#		--result_dir 'trained_models/mlclf/' \
#		--n_train_epoch 50 \
#		--n_repeats 1 \
#		--eval_test_at_final_loop_only true \
#		--noisy_val false \
#		--seed 100
#		done
#	done 
#done

# Hyperparam tuning for tomato dataset
#for batch_size in 32 64 128 
#do
#	for lr in 5e-5 5e-4 5e-3 5e-2
#	do
#		python3 -m training.train_mlclf \
#		--dataset tomato \
#		--lr $lr \
#		--noise_rate 0.3 \
#		--noise_type 'symmetric' \
#		--img_encoder 'resnet50' \
#		--batch_size $batch_size \
#		--clf_name mlclf \
#		--num_workers 2 \
#		--result_dir 'trained_models/mlclf/' \
#		--n_train_epoch 20 \
#		--n_repeats 1 \
#		--eval_test_at_final_loop_only true \
#		--noisy_val false \
#		--seed 100
#	done 
#done


# For coco
# for noise_type in 'pairflip'  
# do
# 	for img_encoder in 'resnet50' 
# 	do
# 		for noise in  '0.5' '0.3'
# 		do
# 		python3 -m training.train_mlclf \
# 		--dataset coco \
# 		--lr 5e-5 \
# 		--noise_rate $noise \
# 		--noise_type $noise_type \
# 		--img_encoder $img_encoder \
# 		--batch_size 128 \
# 		--clf_name mlclf \
# 		--num_workers 4 \
# 		--result_dir 'trained_models/mlclf/' \
# 		--n_train_epoch 30 \
# 		--n_repeats 1 \
# 		--eval_test_at_final_loop_only true \
# 		--noisy_val false \
# 		--seed 100
# 		done
# 	done 
# done


for noise_type in 'symmetric' 
do
	for img_encoder in 'levit' 'resnet50' 
	do
		for noise in '0.0' '0.4' '0.5' '0.3'
		do
		python3 -m training.train_mlclf \
		--dataset coco \
		--lr 5e-5 \
		--noise_rate $noise \
		--noise_type $noise_type \
		--img_encoder $img_encoder \
		--batch_size 128 \
		--clf_name mlclf \
		--num_workers 4 \
		--result_dir 'trained_models/mlclf/' \
		--n_train_epoch 30 \
		--n_repeats 1 \
		--eval_test_at_final_loop_only true \
		--noisy_val false \
		--seed 100
		done
	done 
done