#!/usr/bin/env bash

# tomato
for noise_type in 'symmetric' 
do
	for img_encoder in 'resnet50' 
	do
		for noise in 0.4 0.3 0.5 0
		do 
		python3 -m training.train_hlc \
		--dataset tomato \
		--lr 1e-5 \
		--noise_rate $noise \
		--noise_type $noise_type \
		--img_encoder $img_encoder \
		--batch_size 32 \
		--clf_name hlc \
		--num_workers 4 \
		--result_dir 'trained_models/hlc/' \
		--delta 0.45 \
		--epoch_update_start 5 \
		--noisy_val true \
		--n_train_epoch 40 \
		--n_repeats 5 \
		--eval_test_at_final_loop_only true \
		--noisy_val false \
		--seed 100 
		done
	done
done

for noise_type in 'pairflip'  
do
	for img_encoder in 'resnet50' 
	do
		for noise in 0.4 0.3 0.5  
		do 
		python3 -m training.train_hlc \
		--dataset tomato \
		--lr 1e-5 \
		--noise_rate $noise \
		--noise_type $noise_type \
		--img_encoder $img_encoder \
		--batch_size 32 \
		--clf_name hlc \
		--num_workers 4 \
		--result_dir 'trained_models/hlc/' \
		--delta 0.45 \
		--epoch_update_start 5 \
		--noisy_val true \
		--n_train_epoch 40 \
		--n_repeats 5 \
		--eval_test_at_final_loop_only true \
		--noisy_val false \
		--seed 100 
		done
	done
done