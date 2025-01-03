#!/usr/bin/env bash
  
# For coco
for noise_type in 'symmetric' 'pairflip'
do
	for noise in '0.4' '0.5' '0.0'
	do
		if [ "$noise" == "0.0" ] && [ "$noise_type" == "pairflip" ]
		then
			continue 1
		fi

		for img_encoder in 'resnet50' 'levit' 
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
 