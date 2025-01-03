#!/usr/bin/env bash

for noise_type in 'symmetric' 'pairflip'
do
	for noise in '0.4' # '0.5' '0.0'
	do
		if [ "$noise" == "0.0" ] && [ "$noise_type" == "pairflip" ]
		then
			continue 1
		fi
 
		python3 -m training.train_hlc \
		--dataset voc2012 \
		--noise_rate $noise \
		--noise_type $noise_type \
		--img_encoder 'resnet50' \
		--clf_name hlc \
		--num_workers 4 \
		--result_dir 'trained_models/hlc/' \
		--delta 0.45 \
		--epoch_update_start 5 \
		--noisy_val false \
		--n_train_epoch 20 \
		--n_repeats 1 \
		--eval_test_at_final_loop_only true \
		--noisy_val false \
		--seed 100 
	done 
done

