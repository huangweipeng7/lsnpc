#!/usr/bin/env bash

# voc2007
for noise_type in 'symmetric' 
do
	for noise in 0.4 0.3 0.5 0
	do 
	python3 -m training.train_hlc \
	--dataset voc2007 \
	--noise_rate $noise \
	--noise_type $noise_type \
	--img_encoder 'resnet50' \
	--batch_size 128 \
	--clf_name addgcn \
	--num_workers 4 \
	--result_dir 'trained_models/addgcn/' \
	--n_train_epoch 20 \
	--n_repeats 1 \
	--seed 100 \
	--eval_test_at_final_loop_only true \
	--noisy_val false \
	--epoch_update_start 10000	# turn off HLC
	done
done


for noise_type in 'pairflip'  
do
	for noise in 0.4 0.3 0.5  
	do 
	python3 -m training.train_hlc \
	--dataset voc2007 \
	--noise_rate $noise \
	--noise_type $noise_type \
	--img_encoder 'resnet50' \
	--batch_size 128 \
	--clf_name addgcn \
	--num_workers 4 \
	--result_dir 'trained_models/addgcn/' \
	--n_train_epoch 20 \
	--n_repeats 1 \
	--seed 100 \
	--eval_test_at_final_loop_only true \
	--noisy_val false \
	--epoch_update_start 10000	# turn off HLC
	done
done